"""
document_processor.py
PDF/TXT 解析 + 文本分段模块
支持：PDF 文本层提取、OCR 回退、自定义文本分段
"""

import os
import re
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pdfplumber
from dotenv import load_dotenv
from loguru import logger
from PIL import Image

load_dotenv()

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 512))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 100))
OCR_BACKEND = os.getenv("OCR_BACKEND", "auto").strip().lower()
OCR_LANG = os.getenv("OCR_LANG", "ch")
OCR_MIN_TEXT_CHARS = int(os.getenv("OCR_MIN_TEXT_CHARS", 80))
OCR_EMPTY_PAGE_RATIO = float(os.getenv("OCR_EMPTY_PAGE_RATIO", 0.5))
OCR_RENDER_SCALE = float(os.getenv("OCR_RENDER_SCALE", 2.0))


class RecursiveCharacterTextSplitter:
    """本地实现的文本分段器（增强版：法律/中文友好）。"""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 100,
        separators=None,
        keep_separator: bool = True,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", "。", "；", "！", "？", "，", " "]
        self.keep_separator = keep_separator

    def split_text(self, text: str) -> list[str]:
        text = text.strip()
        if not text:
            return []

        articles = self._split_by_article(text)
        if len(articles) == 1:
            return self._split_recursive(text)

        final_chunks: list[str] = []
        for art in articles:
            art = art.strip()
            if not art:
                continue

            if len(art) <= self.chunk_size:
                final_chunks.append(art)
            else:
                sub_chunks = self._split_recursive(art)
                sub_chunks = self._merge_chunks(sub_chunks)
                final_chunks.extend(sub_chunks)

        return final_chunks

    def split_text_for_other(self, text: str) -> list[str]:
        text = text.strip()
        if not text:
            return []

        articles = self._split_by_article(text)
        if len(articles) > 1:
            return self._section_chunks(articles)

        numbered = self._split_by_chinese_numbers(text)
        if len(numbered) > 1:
            return self._section_chunks(numbered)

        return self._split_recursive(text)

    def _section_chunks(self, sections: list[str]) -> list[str]:
        final: list[str] = []
        for sec in sections:
            sec = sec.strip()
            if not sec:
                continue
            if len(sec) <= self.chunk_size:
                final.append(sec)
            else:
                sub = self._split_recursive(sec)
                sub = self._merge_chunks(sub)
                final.extend(sub)
        return final

    def _split_by_chinese_numbers(self, text: str) -> list[str]:
        pattern = re.compile(
            r"(?:^|\n)\s*"
            r"(?:（\s*[一二三四五六七八九十]+\s*）|\(\s*[一二三四五六七八九十]+\s*\))"
            r"|"
            r"(?:^|\n)\s*[一二三四五六七八九十]+(?:[一二三四五六七八九十])?\s*[、.．]",
            re.MULTILINE,
        )
        matches = list(pattern.finditer(text))
        if len(matches) < 2:
            return [text]

        sections = []
        if matches[0].start() > 0:
            preamble = text[:matches[0].start()].strip()
            if preamble:
                sections.append(preamble)

        for i, m in enumerate(matches):
            start = m.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            part = text[start:end].strip()
            if part:
                sections.append(part)

        return sections

    def _split_by_article(self, text: str) -> list[str]:
        pattern = r"(第[零一二三四五六七八九十百千万\d]+条[^第]*)"
        matches = re.findall(pattern, text)
        if not matches:
            return [text]
        return [m.strip() for m in matches if m.strip()]

    def _split_recursive(self, text: str) -> list[str]:
        text = text.strip()
        if not text:
            return []

        if len(text) <= self.chunk_size:
            return [text]

        for separator in self.separators:
            if not separator:
                continue

            if separator in text:
                parts = text.split(separator)
                if len(parts) == 1:
                    continue

                chunks = []
                for i, part in enumerate(parts):
                    if not part.strip():
                        continue

                    piece = part
                    if self.keep_separator and i < len(parts) - 1:
                        piece += separator

                    piece = piece.strip()
                    if len(piece) >= len(text):
                        return self._split_by_char(text)

                    chunks.extend(self._split_recursive(piece))

                if len(chunks) > 1:
                    return self._merge_chunks(chunks)

        return self._split_by_char(text)

    def _split_by_char(self, text: str) -> list[str]:
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk = text[start:end]
            chunks.append(chunk)
            start += self.chunk_size - self.chunk_overlap
        return chunks

    def _merge_chunks(self, chunks: list[str]) -> list[str]:
        merged = []
        current = ""
        for chunk in chunks:
            if not current:
                current = chunk
                continue
            if len(current) + len(chunk) <= self.chunk_size:
                current += chunk
            else:
                merged.append(current)
                overlap = current[-self.chunk_overlap :] if self.chunk_overlap > 0 else ""
                current = overlap + chunk
        if current:
            merged.append(current)
        return merged


class DocumentProcessor:
    """PDF/TXT 解析与文本分段。"""

    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.ocr_backend = OCR_BACKEND
        self.ocr_lang = OCR_LANG
        self.ocr_min_text_chars = OCR_MIN_TEXT_CHARS
        self.ocr_empty_page_ratio = OCR_EMPTY_PAGE_RATIO
        self.ocr_render_scale = OCR_RENDER_SCALE
        self._paddle_ocr = None
        self._docling_converter = None
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "；", "！", "？", "，", " "],
            keep_separator=True,
        )

    def extract_text_from_pdf(self, pdf_path: str) -> dict[str, Any]:
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF文件不存在: {pdf_path}")

        pages, full_text_parts = self._extract_pdf_text_layer(pdf_path)
        if self._should_use_ocr(pages):
            logger.info(f"PDF文本层不足，尝试OCR: {path.name} (backend={self.ocr_backend})")
            ocr_result = self._extract_text_from_pdf_with_ocr(pdf_path)
            if ocr_result["full_text"].strip():
                return ocr_result
            logger.warning(f"OCR未提取到有效文本，回退文本层结果: {path.name}")

        return {
            "title": path.stem,
            "file_path": str(path.resolve()),
            "pages": pages,
            "full_text": "\n\n".join(full_text_parts),
            "extraction_method": "pdf_text",
        }

    def _extract_pdf_text_layer(self, pdf_path: str) -> tuple[list[dict[str, Any]], list[str]]:
        path = Path(pdf_path)
        pages: list[dict[str, Any]] = []
        full_text_parts: list[str] = []

        with pdfplumber.open(pdf_path) as pdf:
            logger.info(f"解析PDF文本层: {path.name}，共{len(pdf.pages)}页")
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                text = self._clean_text(text)
                pages.append({"page_num": i + 1, "text": text})
                if text:
                    full_text_parts.append(text)

        return pages, full_text_parts

    def _should_use_ocr(self, pages: list[dict[str, Any]]) -> bool:
        if self.ocr_backend == "none":
            return False
        if not pages:
            return True

        empty_pages = sum(1 for page in pages if not page["text"].strip())
        total_chars = sum(len(page["text"].strip()) for page in pages)
        empty_ratio = empty_pages / len(pages)
        return total_chars < self.ocr_min_text_chars or empty_ratio >= self.ocr_empty_page_ratio

    def _extract_text_from_pdf_with_ocr(self, pdf_path: str) -> dict[str, Any]:
        backends = self._resolve_ocr_backends()
        last_error = None

        for backend in backends:
            try:
                if backend == "docling":
                    result = self._extract_text_with_docling(pdf_path)
                elif backend == "paddleocr":
                    result = self._extract_text_with_paddleocr(pdf_path)
                else:
                    continue

                if result["full_text"].strip():
                    return result
            except Exception as exc:
                last_error = exc
                logger.warning(f"OCR backend {backend} 失败: {exc}")

        if last_error is not None:
            logger.warning(f"全部OCR backend失败，保留原始文本层结果: {last_error}")

        path = Path(pdf_path)
        return {
            "title": path.stem,
            "file_path": str(path.resolve()),
            "pages": [],
            "full_text": "",
            "extraction_method": "ocr_failed",
        }

    def _resolve_ocr_backends(self) -> list[str]:
        if self.ocr_backend == "auto":
            return ["docling", "paddleocr"]
        if self.ocr_backend in {"docling", "paddleocr"}:
            return [self.ocr_backend]
        return []

    def _extract_text_with_docling(self, pdf_path: str) -> dict[str, Any]:
        converter = self._get_docling_converter()
        result = converter.convert(pdf_path)
        document = getattr(result, "document", result)
        markdown = ""

        export_markdown = getattr(document, "export_to_markdown", None)
        if callable(export_markdown):
            markdown = export_markdown() or ""
        elif hasattr(document, "text"):
            markdown = document.text or ""
        else:
            markdown = str(document)

        full_text = self._clean_text(markdown)
        path = Path(pdf_path)
        logger.info(f"Docling OCR完成: {path.name}")
        return {
            "title": path.stem,
            "file_path": str(path.resolve()),
            "pages": [{"page_num": 1, "text": full_text}] if full_text else [],
            "full_text": full_text,
            "extraction_method": "docling",
        }

    def _extract_text_with_paddleocr(self, pdf_path: str) -> dict[str, Any]:
        import fitz

        ocr = self._get_paddle_ocr()
        path = Path(pdf_path)
        pages: list[dict[str, Any]] = []
        full_text_parts: list[str] = []

        with fitz.open(pdf_path) as pdf:
            logger.info(f"PaddleOCR解析PDF: {path.name}，共{pdf.page_count}页")
            for page_index in range(pdf.page_count):
                page = pdf.load_page(page_index)
                matrix = fitz.Matrix(self.ocr_render_scale, self.ocr_render_scale)
                pix = page.get_pixmap(matrix=matrix, alpha=False)
                image = Image.open(BytesIO(pix.tobytes("png"))).convert("RGB")
                text = self._ocr_image_with_paddle(ocr, image)
                text = self._clean_text(text)
                pages.append({"page_num": page_index + 1, "text": text})
                if text:
                    full_text_parts.append(text)

        return {
            "title": path.stem,
            "file_path": str(path.resolve()),
            "pages": pages,
            "full_text": "\n\n".join(full_text_parts),
            "extraction_method": "paddleocr",
        }

    def _ocr_image_with_paddle(self, ocr: Any, image: Image.Image) -> str:
        result = ocr.ocr(np.array(image), cls=True)
        lines: list[str] = []
        for page_result in result or []:
            for item in page_result or []:
                if len(item) >= 2 and item[1]:
                    lines.append(str(item[1][0]).strip())
        return "\n".join(line for line in lines if line)

    def _get_paddle_ocr(self):
        if self._paddle_ocr is None:
            from paddleocr import PaddleOCR

            self._paddle_ocr = PaddleOCR(use_angle_cls=True, lang=self.ocr_lang)
        return self._paddle_ocr

    def _get_docling_converter(self):
        if self._docling_converter is None:
            from docling.document_converter import DocumentConverter

            self._docling_converter = DocumentConverter()
        return self._docling_converter

    def _clean_text(self, text: str) -> str:
        text = text.replace("\x00", " ")
        lines = [line.strip() for line in text.splitlines()]
        cleaned_lines = []
        prev_empty = False
        for line in lines:
            if not line:
                if not prev_empty:
                    cleaned_lines.append("")
                prev_empty = True
            else:
                cleaned_lines.append(line)
                prev_empty = False
        return "\n".join(cleaned_lines).strip()

    def split_into_chunks(self, doc_info: dict[str, Any], doc_type: str = "law") -> list[dict[str, Any]]:
        full_text = doc_info["full_text"]

        if doc_type == "case":
            chunks_text = self.splitter.split_text(full_text)
        elif doc_type == "other":
            chunks_text = self.splitter.split_text_for_other(full_text)
        else:
            chunks_text = self.splitter.split_text(full_text)

        law_name, effective_date = self._extract_law_name_and_date(doc_info["title"], doc_type)
        authority = self._extract_authority(doc_info["title"], doc_info.get("file_path", ""))

        chunks = []
        for idx, chunk_text in enumerate(chunks_text):
            article_match = re.search(r"第[零一二三四五六七八九十百\d]+条", chunk_text)
            article_num = article_match.group(0) if article_match else ""

            chunks.append({
                "chunk_id": f"{doc_info['title']}_chunk_{idx:04d}",
                "text": chunk_text,
                "source": doc_info["title"],
                "file_path": doc_info["file_path"],
                "article_num": article_num,
                "chunk_index": idx,
                "doc_type": doc_type,
                "law_name": law_name,
                "authority": authority,
                "effective_date": effective_date,
                "status": "有效",  # 多版本场景由 resolve_version_status 修正
            })

        logger.info(
            f"文档 [{doc_info['title']}] 分段完成，共 {len(chunks)} 个chunk，提取方式={doc_info.get('extraction_method', 'unknown')}"
        )
        return chunks

    @staticmethod
    def _extract_date_from_title(title: str) -> str:
        """从文件名提取日期后缀 _YYYYMMDD → YYYY-MM-DD"""
        m = re.search(r"_(\d{4})(\d{2})(\d{2})$", title)
        if m:
            return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
        return ""

    def _extract_law_name_and_date(self, title: str, doc_type: str) -> tuple:
        """从 title 同时提取法规名和生效日期。返回 (law_name, effective_date)"""
        if doc_type != "law":
            return title, self._extract_date_from_title(title)
        law_name = re.sub(r"_\d{8}$", "", title)
        date_str = self._extract_date_from_title(title)
        return (law_name or title, date_str)

    @staticmethod
    def resolve_version_status(all_chunks: list) -> list:
        """同名法规多版本状态判定: 最新日期 → 有效，其余 → 已修订。"""
        from collections import defaultdict
        groups: dict[str, list] = defaultdict(list)
        for c in all_chunks:
            groups[c.get("law_name", "")].append(c)
        for group in groups.values():
            if not group:
                continue
            group.sort(key=lambda c: c.get("effective_date", ""), reverse=True)
            latest_date = group[0].get("effective_date", "")
            for c in group:
                c["status"] = "有效" if c.get("effective_date", "") == latest_date else "已修订"
        return all_chunks

    def _extract_authority(self, title: str, file_path: str) -> str:
        for kw in ["关于", "办公室关于"]:
            if kw in title:
                prefix = title.split(kw)[0].strip()
                prefix = re.sub(r"^(国家金融监督管理总局|中国银保监会|中国保监会|中国银监会)", "", prefix)
                prefix = re.sub(r"(办公室|秘书处)$", "", prefix)
                if len(prefix) >= 4:
                    return prefix

        if file_path:
            path_parts = file_path.replace("\\", "/").split("/")
            for part in path_parts:
                if part.endswith("监管局") or part.endswith("监管分局"):
                    return part

        return ""

    def process_pdf(self, pdf_path: str, doc_type: str = "law") -> list[dict[str, Any]]:
        doc_info = self.extract_text_from_pdf(pdf_path)
        return self.split_into_chunks(doc_info, doc_type=doc_type)

    def extract_text_from_txt(self, txt_path: str) -> dict[str, Any]:
        path = Path(txt_path)
        if not path.exists():
            raise FileNotFoundError(f"文本文件不存在: {txt_path}")

        with path.open("r", encoding="utf-8", errors="ignore") as f:
            text = f.read().strip()

        text = self._clean_text(text)
        return {
            "title": path.stem,
            "file_path": str(path.resolve()),
            "pages": [{"page_num": 1, "text": text}],
            "full_text": text,
            "extraction_method": "txt",
        }

    def process_txt(self, txt_path: str, doc_type: str = "law") -> list[dict[str, Any]]:
        doc_info = self.extract_text_from_txt(txt_path)
        return self.split_into_chunks(doc_info, doc_type=doc_type)

    def process_file(self, file_path: str, doc_type: str = "law") -> list[dict[str, Any]]:
        suffix = Path(file_path).suffix.lower()
        if suffix == ".pdf":
            return self.process_pdf(file_path, doc_type=doc_type)
        if suffix == ".txt":
            return self.process_txt(txt_path=file_path, doc_type=doc_type)
        raise ValueError(f"不支持的文件类型: {suffix}，仅支持PDF和TXT")

    def process_directory(self, dir_path: str) -> list[dict[str, Any]]:
        all_chunks = []
        files = list(Path(dir_path).glob("**/*"))
        supported_files = [f for f in files if f.suffix.lower() in {".pdf", ".txt"}]
        logger.info(f"发现 {len(supported_files)} 个支持的文件（PDF/TXT）")
        for file_path in supported_files:
            try:
                chunks = self.process_file(str(file_path))
                all_chunks.extend(chunks)
            except Exception as e:
                logger.error(f"处理 {file_path.name} 失败: {e}")
        return all_chunks

"""
document_processor.py
PDF解析 + 文本分段模块
支持：PDF文本层提取（pdfplumber）、章节层级保留、自定义文本分段
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Any

import pdfplumber
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 512))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 100))


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
        # ⚠️ 不再把 "第" 作为普通分隔符，避免破坏“第XX条”结构
        self.separators = separators or ["\n\n", "\n", "。", "；", "！", "？", "，", " "]
        self.keep_separator = keep_separator

    # ===================== 核心：对外接口 =====================
    def split_text(self, text: str) -> list[str]:
        text = text.strip()
        if not text:
            return []

        # 1️⃣ 强制优先：按“第XX条”切分（命中则绝不跨条合并）
        articles = self._split_by_article(text)

        # 未命中条文结构 → 走原递归
        if len(articles) == 1:
            return self._split_recursive(text)

        # 2️⃣ 条内再切分，并且“每条独立合并”，避免跨条拼接
        final_chunks: list[str] = []
        for art in articles:
            art = art.strip()
            if not art:
                continue

            if len(art) <= self.chunk_size:
                # 单条较短，直接作为一个chunk（仍保留条头）
                final_chunks.append(art)
            else:
                # 条内递归切分
                sub_chunks = self._split_recursive(art)
                # ⚠️ 关键：只在“条内”做merge，不与其他条混合
                sub_chunks = self._merge_chunks(sub_chunks)
                final_chunks.extend(sub_chunks)

        return final_chunks

    def split_text_for_other(self, text: str) -> list[str]:
        """其它资料切分：第X条 → 中文序号 → 递归，三级优先级尝试。"""
        text = text.strip()
        if not text:
            return []

        # 1. 先尝试"第X条"切分（管理办法/实施细则类）
        articles = self._split_by_article(text)
        if len(articles) > 1:
            return self._section_chunks(articles)

        # 2. 尝试按中文序号切分（通知/指导意见类：一、二、三、...）
        numbered = self._split_by_chinese_numbers(text)
        if len(numbered) > 1:
            return self._section_chunks(numbered)

        # 3. 兜底：纯递归切分（公告/模板类）
        return self._split_recursive(text)

    def _section_chunks(self, sections: list[str]) -> list[str]:
        """每个结构段落内部独立切分+合并，避免跨段落拼接。"""
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
        """按中文序号（一、二、三、... 或（一）（二）...）切分全文。"""
        # 匹配行首中文数字 + 标点：一、 二、 （一） (一) 十二、
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

    # ===================== 按条切分 =====================
    def _split_by_article(self, text: str) -> list[str]:
        """按“第XX条”切分，保留条头"""
        pattern = r"(第[零一二三四五六七八九十百千万\d]+条[^第]*)"
        matches = re.findall(pattern, text)
        if not matches:
            return [text]
        return [m.strip() for m in matches if m.strip()]

    # ===================== 原递归逻辑（轻微改造） =====================
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

                    # 防止递归不收敛
                    if len(piece) >= len(text):
                        return self._split_by_char(text)

                    chunks.extend(self._split_recursive(piece))

                if len(chunks) > 1:
                    return self._merge_chunks(chunks)

        return self._split_by_char(text)

    # ===================== 底层工具 =====================
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
    """PDF解析与文本分段"""

    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "；", "！", "？", "，", " "],
            keep_separator=True,
        )

    def extract_text_from_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        提取PDF全文，保留页码信息
        Returns:
            {
                "title": str,       # 文件名（作为文档标题）
                "pages": [          # 按页的文本列表
                    {"page_num": int, "text": str}
                ],
                "full_text": str    # 合并全文
            }
        """
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF文件不存在: {pdf_path}")

        pages = []
        full_text_parts = []

        with pdfplumber.open(pdf_path) as pdf:
            logger.info(f"解析PDF: {path.name}，共{len(pdf.pages)}页")
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text:
                    text = self._clean_text(text)
                    pages.append({"page_num": i + 1, "text": text})
                    full_text_parts.append(text)

        return {
            "title": path.stem,
            "file_path": str(path.resolve()),
            "pages": pages,
            "full_text": "\n\n".join(full_text_parts),
        }

    def _clean_text(self, text: str) -> str:
        """清理文本：去除多余空白，修正常见OCR错误"""
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

    def split_into_chunks(self, doc_info: Dict[str, Any], doc_type: str = "law") -> List[Dict[str, Any]]:
        """
        将文档全文分段，返回带元数据的chunk列表
        每个chunk包含：text, source, chunk_id, article_num(条文号，如有), doc_type
        """
        full_text = doc_info["full_text"]

        if doc_type == "case":
            chunks_text = self.splitter.split_text(full_text)
        elif doc_type == "other":
            chunks_text = self.splitter.split_text_for_other(full_text)
        else:
            chunks_text = self.splitter.split_text(full_text)

        chunks = []
        for idx, chunk_text in enumerate(chunks_text):
            article_match = re.search(r"第[零一二三四五六七八九十百\d]+条", chunk_text)
            article_num = article_match.group(0) if article_match else ""

            law_name = self._extract_law_name(doc_info["title"], doc_type)
            authority = self._extract_authority(doc_info["title"], doc_info.get("file_path", ""))

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
            })

        logger.info(f"文档 [{doc_info['title']}] 分段完成，共 {len(chunks)} 个chunk")
    def _extract_law_name(self, title: str, doc_type: str) -> str:
        """
        从文件名提取法律名称。
        "中华人民共和国公司法_20131228" → "中华人民共和国公司法"
        案例和其它资料返回原始 title。
        """
        if doc_type != "law":
            return title
        # 去掉末尾的 _日期 后缀（如 _20131228、_20250424）
        cleaned = re.sub(r"_\d{8}$", "", title)
        return cleaned or title

    def _extract_authority(self, title: str, file_path: str) -> str:
        """
        从文件名或路径提取发布机构。
        "上海银保监局关于印发《...》的通知"  →  "上海银保监局"
        "中国银保监会南通监管分局关于..."    →  "南通监管分局"
        无明确机构时从目录路径推断，兜底返回空字符串。
        """
        # 策略一：取"关于"之前的文本作为机构名
        for kw in ["关于", "办公室关于"]:
            if kw in title:
                prefix = title.split(kw)[0].strip()
                # 清理常见前缀/后缀
                prefix = re.sub(r"^(国家金融监督管理总局|中国银保监会|中国保监会|中国银监会)", "", prefix)
                prefix = re.sub(r"(办公室|秘书处)$", "", prefix)
                if len(prefix) >= 4:
                    return prefix

        # 策略二：从目录路径推断（如 "金融监督管理局/上海监管局/..."）
        if file_path:
            path_parts = file_path.replace("\\", "/").split("/")
            for i, part in enumerate(path_parts):
                if part.endswith("监管局") or part.endswith("监管分局"):
                    return part

        return ""

    def process_pdf(self, pdf_path: str, doc_type: str = "law") -> List[Dict[str, Any]]:
        doc_info = self.extract_text_from_pdf(pdf_path)
        return self.split_into_chunks(doc_info, doc_type=doc_type)

    def extract_text_from_txt(self, txt_path: str) -> Dict[str, Any]:
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
        }

    def process_txt(self, txt_path: str, doc_type: str = "law") -> List[Dict[str, Any]]:
        doc_info = self.extract_text_from_txt(txt_path)
        return self.split_into_chunks(doc_info, doc_type=doc_type)

    def process_file(self, file_path: str, doc_type: str = "law") -> List[Dict[str, Any]]:
        suffix = Path(file_path).suffix.lower()
        if suffix == ".pdf":
            return self.process_pdf(file_path, doc_type=doc_type)
        if suffix == ".txt":
            return self.process_txt(txt_path=file_path, doc_type=doc_type)
        raise ValueError(f"不支持的文件类型: {suffix}，仅支持PDF和TXT")

    def process_directory(self, dir_path: str) -> List[Dict[str, Any]]:
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
      
    
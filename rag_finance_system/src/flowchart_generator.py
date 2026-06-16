"""
flowchart_generator.py
图片/文本 → Mermaid 流程图生成模块

双路径架构：
- 路径A（首选）：QwenVL 多模态 LLM 直接理解图片 → Mermaid 语法
- 路径B（回退）：PaddleOCR/Docling 提取文本 → 文本 LLM → Mermaid 语法
"""

import base64
import mimetypes
import re

from loguru import logger

# QwenVLAPILLM 仅用于 isinstance 类型判断，模块级加载可在依赖缺失时降级
try:
    from rag_finance_system.src.llm import QwenVLAPILLM
except ImportError as e:
    logger.warning(f"QwenVLAPILLM 导入失败（多模态 LLM 类型判断不可用）: {e}")
    logger.warning("请确保 openai 包已安装: pip install openai")
    QwenVLAPILLM = None

# ── Prompt 定义 ──

MULTIMODAL_SYSTEM_PROMPT = """你是一个金融法规流程分析专家。请分析这张图片中描述的流程/步骤/程序。

要求：
1. 识别图片中所有步骤、决策点、条件分支
2. 用 Mermaid flowchart 语法生成流程图
3. 步骤名称用中文，保持简洁（不超过15字）
4. 条件分支用菱形节点
5. 如有明确的流程方向（箭头/序号），按原图方向组织
6. 只输出 Mermaid 代码块，不要解释

如果图片中没有可识别的流程或步骤，请输出：
```mermaid
无流程
```

输出格式示例：
```mermaid
flowchart TD
    A[申请人提交材料] --> B{材料是否齐全}
    B -->|是| C[受理申请]
    B -->|否| D[退回补充材料]
    D --> A
    C --> E[审查审批]
    E --> F{是否批准}
    F -->|批准| G[颁发许可证]
    F -->|不批准| H[书面通知并说明理由]
```"""

TEXT_SYSTEM_PROMPT = """你是一个金融法规流程分析专家。请从以下法规文本中识别并提取流程/步骤/程序，生成 Mermaid 流程图。

要求：
1. 从文本中识别所有程序性步骤（如：申请→受理→审查→决定）
2. 识别条件分支（如"符合XX条件的，...；不符合的，..."）
3. 用 Mermaid flowchart 语法生成流程图
4. 步骤名称用原文关键词，保持简洁（不超过15字）
5. 只输出 Mermaid 代码块，不要解释

如果文本中没有可识别的流程或步骤，请输出：
```mermaid
无流程
```

输出格式示例：
```mermaid
flowchart TD
    A[申请] --> B[受理]
    B --> C[审查]
    C --> D{是否符合条件}
    D -->|是| E[批准]
    D -->|否| F[驳回并说明理由]
```"""


class FlowchartGenerator:
    """图片/文本 → Mermaid 流程图生成器。"""

    def __init__(self, multimodal_llm=None, text_llm=None, ocr_processor=None):
        self.multimodal_llm = multimodal_llm
        self.text_llm = text_llm
        self.ocr_processor = ocr_processor

    def generate_from_image(
        self,
        image_path: str | None = None,
        image_base64: str | None = None,
    ) -> dict:
        """路径A：多模态 LLM 直接从图片生成 Mermaid。

        Args:
            image_path: 图片文件路径
            image_base64: 图片 base64 编码（与 image_path 二选一）

        Returns:
            {"mermaid": str, "source": "multimodal", "raw_text": str, "success": bool, "error": str|None}
        """
        if self.multimodal_llm is None:
            return self._fallback_to_ocr(image_path, image_base64)

        try:
            # 构建 base64 data URI
            if image_path:
                data_uri = self._encode_image_to_data_uri(image_path)
            elif image_base64:
                # image_base64 可能已经是 data URI 或纯 base64
                if image_base64.startswith("data:"):
                    data_uri = image_base64
                else:
                    data_uri = f"data:image/png;base64,{image_base64}"
            else:
                return {"mermaid": "", "source": "multimodal", "raw_text": "", "success": False, "error": "未提供图片"}

            # 构建多模态消息
            messages = [
                {"role": "system", "content": MULTIMODAL_SYSTEM_PROMPT},
            ]
            if QwenVLAPILLM is not None and isinstance(self.multimodal_llm, QwenVLAPILLM):
                user_content = [
                    {"type": "image_url", "image_url": {"url": data_uri}},
                    {"type": "text", "text": "请分析这张图片中的流程/步骤，生成 Mermaid 流程图。"},
                ]
                messages.append({"role": "user", "content": user_content})
            else:
                # 其他多模态 LLM 也用类似格式
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_uri}},
                        {"type": "text", "text": "请分析这张图片中的流程/步骤，生成 Mermaid 流程图。"},
                    ],
                })

            raw_text = self.multimodal_llm.generate(messages, max_new_tokens=2048, temperature=0.1)
            mermaid = self._extract_mermaid_block(raw_text)

            if not mermaid or mermaid == "无流程":
                return {
                    "mermaid": "",
                    "source": "multimodal",
                    "raw_text": raw_text,
                    "success": False,
                    "error": "图片未包含可识别的流程步骤",
                }

            return {"mermaid": mermaid, "source": "multimodal", "raw_text": raw_text, "success": True, "error": None}

        except Exception as e:
            logger.warning(f"多模态路径失败: {e}，回退 OCR 路径")
            return self._fallback_to_ocr(image_path, image_base64)

    def generate_from_text(self, text: str) -> dict:
        """路径B：从文本生成 Mermaid。

        Args:
            text: 法规文本（来自 OCR 或用户直接输入）

        Returns:
            {"mermaid": str, "source": "ocr+llm"|"text", "raw_text": str, "ocr_text": str|None, "success": bool, "error": str|None}
        """
        if len(text.strip()) < 50:
            return {
                "mermaid": "",
                "source": "text",
                "raw_text": "",
                "ocr_text": None,
                "success": False,
                "error": "输入文本过短（少于50字符），可能不含流程信息",
            }

        if self.text_llm is None:
            return {"mermaid": "", "source": "text", "raw_text": "", "ocr_text": None, "success": False, "error": "文本 LLM 不可用"}

        try:
            messages = [
                {"role": "system", "content": TEXT_SYSTEM_PROMPT},
                {"role": "user", "content": f"请从以下文本中提取流程步骤并生成 Mermaid 流程图：\n\n{text}"},
            ]
            raw_text = self.text_llm.generate(messages, max_new_tokens=2048, temperature=0.1)
            mermaid = self._extract_mermaid_block(raw_text)

            if not mermaid or mermaid == "无流程":
                return {
                    "mermaid": "",
                    "source": "text",
                    "raw_text": raw_text,
                    "ocr_text": None,
                    "success": False,
                    "error": "文本中未包含可识别的流程步骤",
                }

            return {"mermaid": mermaid, "source": "text", "raw_text": raw_text, "ocr_text": None, "success": True, "error": None}

        except Exception as e:
            logger.error(f"文本路径生成失败: {e}")
            return {"mermaid": "", "source": "text", "raw_text": "", "ocr_text": None, "success": False, "error": str(e)}

    def generate_from_image_via_ocr(
        self,
        image_path: str | None = None,
        image_base64: str | None = None,
    ) -> dict:
        """路径B（从图片）：OCR 提取文本 → 文本 LLM → Mermaid。"""
        if self.ocr_processor is None:
            return {"mermaid": "", "source": "ocr+llm", "raw_text": "", "ocr_text": None, "success": False, "error": "OCR 处理器不可用"}

        try:
            if image_path:
                ocr_text = self.ocr_processor.extract_text_from_image(image_path)
            elif image_base64:
                # base64 图片需要先保存为临时文件再 OCR
                import tempfile
                from pathlib import Path

                # 解码 base64
                if image_base64.startswith("data:"):
                    # 提取纯 base64 部分
                    _, b64_data = image_base64.split(",", 1)
                else:
                    b64_data = image_base64

                img_bytes = base64.b64decode(b64_data)
                # 尝试推断格式
                suffix = ".png"
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                    tmp.write(img_bytes)
                    tmp_path = tmp.name

                try:
                    ocr_text = self.ocr_processor.extract_text_from_image(tmp_path)
                finally:
                    Path(tmp_path).unlink(missing_ok=True)
            else:
                return {"mermaid": "", "source": "ocr+llm", "raw_text": "", "ocr_text": None, "success": False, "error": "未提供图片"}

            if not ocr_text or len(ocr_text.strip()) < 50:
                return {
                    "mermaid": "",
                    "source": "ocr+llm",
                    "raw_text": "",
                    "ocr_text": ocr_text,
                    "success": False,
                    "error": "OCR 提取文本过少，图片可能不含流程信息",
                }

            result = self.generate_from_text(ocr_text)
            result["source"] = "ocr+llm"
            result["ocr_text"] = ocr_text
            return result

        except Exception as e:
            logger.error(f"OCR 路径失败: {e}")
            return {"mermaid": "", "source": "ocr+llm", "raw_text": "", "ocr_text": None, "success": False, "error": str(e)}

    def _fallback_to_ocr(self, image_path: str | None, image_base64: str | None) -> dict:
        """多模态失败时回退到 OCR + 文本 LLM 路径。"""
        logger.info("多模态路径不可用，回退 OCR + 文本 LLM")
        return self.generate_from_image_via_ocr(image_path, image_base64)

    @staticmethod
    def _encode_image_to_data_uri(image_path: str) -> str:
        """将本地图片编码为 base64 data URI。"""
        mime, _ = mimetypes.guess_type(image_path)
        if not mime or not mime.startswith("image/"):
            mime = "image/png"
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return f"data:{mime};base64,{b64}"

    @staticmethod
    def _extract_mermaid_block(text: str) -> str:
        """从 LLM 输出中提取 ```mermaid ... ``` 代码块。

        如果没有代码块标记，尝试直接检测 Mermaid 语法。
        """
        # 优先提取 ```mermaid ... ``` 代码块
        pattern = r"```mermaid\s*\n(.*?)\n```"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()

        # 回退：提取 ``` ... ``` 代码块（可能没标 mermaid）
        pattern2 = r"```\s*\n(.*?)\n```"
        match2 = re.search(pattern2, text, re.DOTALL)
        if match2:
            content = match2.group(1).strip()
            if content.startswith("flowchart") or content.startswith("graph"):
                return content

        # 最终回退：检查整段文本是否本身就是 Mermaid 语法
        stripped = text.strip()
        if stripped.startswith("flowchart") or stripped.startswith("graph"):
            return stripped

        # 检查是否标记为"无流程"
        if "无流程" in stripped:
            return "无流程"

        return ""

    @staticmethod
    def validate_mermaid(mermaid_str: str) -> bool:
        """校验 Mermaid 语法基本完整性。"""
        if not mermaid_str:
            return False
        # 必须包含 flowchart 或 graph 关键字
        if not (mermaid_str.startswith("flowchart") or mermaid_str.startswith("graph")):
            return False
        # 必须包含至少一个箭头连接
        if "-->" not in mermaid_str and "---" not in mermaid_str:
            return False
        return True

    @staticmethod
    def render_mermaid_html(mermaid_str: str, height: int = 600) -> str:
        """生成包含 Mermaid.js CDN 的 HTML，用于 st.components.v1.html 渲染。"""
        # 转义 HTML 特殊字符
        escaped = mermaid_str.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

        return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
    <style>
        body {{ margin: 0; padding: 20px; background: white; }}
        .mermaid {{ font-family: sans-serif; }}
    </style>
</head>
<body>
    <div class="mermaid">
{escaped}
    </div>
    <script>
        mermaid.initialize({{ startOnLoad: true, theme: 'default', flowchart: {{ useMaxWidth: true, htmlLabels: true, curve: 'basis' }} }});
    </script>
</body>
</html>"""

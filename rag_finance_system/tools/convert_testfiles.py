"""
convert_testfiles.py
将 data/testfiles/ 下所有 .doc/.docx 通过 LibreOffice headless 转为 UTF-8 .txt。
"""
import subprocess
import shutil
import tempfile
from pathlib import Path


BASE = Path(__file__).resolve().parent.parent.parent / "data" / "testfiles"
SOFFICE = r"C:\Program Files\LibreOffice\program\soffice.exe"


def convert_with_libreoffice(src: Path, dst: Path) -> bool:
    """用 LibreOffice headless 转换单个 .doc/.docx → .txt，输出为 UTF-8。"""
    tmpdir = Path(tempfile.mkdtemp())
    tmp_src = tmpdir / src.name
    shutil.copy2(src, tmp_src)

    result = subprocess.run(
        [
            SOFFICE,
            "--headless",
            "--convert-to", "txt:Text",
            "--outdir", str(tmpdir),
            str(tmp_src),
        ],
        capture_output=True, text=True, timeout=120,
    )

    ok = False
    tmp_txt = tmpdir / tmp_src.with_suffix(".txt").name
    if tmp_txt.exists():
        raw = tmp_txt.read_bytes()
        # 检测编码：如果前两字节是 BOM，按 BOM 处理；否则优先 GBK → UTF-8
        if raw[:2] == b"\xff\xfe":
            text = raw.decode("utf-16-le")
        elif raw[:2] == b"\xfe\xff":
            text = raw.decode("utf-16-be")
        elif raw[:3] == b"\xef\xbb\xbf":
            text = raw.decode("utf-8-sig")
        else:
            try:
                text = raw.decode("gbk")
            except Exception:
                text = raw.decode("utf-8", errors="replace")

        text = text.strip()
        if len(text) >= 50:
            dst.write_text(text, encoding="utf-8")
            ok = True

    shutil.rmtree(tmpdir, ignore_errors=True)
    return ok


def main():
    files = list(BASE.rglob("*.doc")) + list(BASE.rglob("*.docx"))
    print(f"找到 {len(files)} 个文件")

    ok, skip, fail = 0, 0, 0
    for src in files:
        dst = src.with_suffix(".txt")

        if dst.exists():
            print(f"  SKIP (已存在): {src.name}")
            skip += 1
            continue

        try:
            if convert_with_libreoffice(src, dst):
                txt_len = len(dst.read_text(encoding="utf-8"))
                print(f"  OK  {src.name}  →  {dst.name}  ({txt_len} chars)")
                ok += 1
            else:
                print(f"  FAIL (内容过短): {src.name}")
                fail += 1
        except Exception as e:
            print(f"  FAIL {src.name}: {e}")
            fail += 1

    print(f"\n完成。成功: {ok}  跳过: {skip}  失败: {fail}")


if __name__ == "__main__":
    main()

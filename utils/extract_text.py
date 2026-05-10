from pathlib import Path

def extract_pdf_text(file_path: Path) -> str:
    """
    提取 PDF 文件中的文本
    """
    try:
        import pdfplumber
    except ImportError:
        raise ImportError("请安装 pdfplumber 库：pip install pdfplumber")

    try:
        with pdfplumber.open(file_path) as pdf:
            full_text = []
            for page in pdf.pages:
                full_text.append(page.extract_text())
            return "\n".join(full_text)
    except Exception as e:
        raise ValueError(f"提取 PDF 文件失败：{e}。文件可能已损坏、加密或不是有效的.pdf文件")


def extract_docx_text(file_path: Path) -> str:
    """
    提取 docx 文件中的文本
    """
    try:
        from docx import Document
    except ImportError:
        raise ImportError("请安装 python-docx 库：pip install python-docx")

    try:
        docx = Document(file_path)
        full_text = []

        # 提取所有段落文本
        for para in docx.paragraphs:
            if para.text.strip():  # 过滤空白行
                full_text.append(para.text)

        # 合并段落文本
        return "\n".join(full_text)
    except Exception as e:
        raise ValueError(f"提取 docx 文件失败：{e}。文件可能已损坏、加密或不是有效的.docx文件")


def extract_doc_text(file_path: Path) -> str:
    """
    提取 doc 文件中的文本
    """
    raise NotImplementedError(
        "不支持 .doc 格式的老版Word文件，请将文件另存为 .docx 格式后重新上传"
    )

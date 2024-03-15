import gradio as gr
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    Docx2txtLoader,
    PDFPlumberLoader,
    TextLoader,
)
from langchain_openai import OpenAI

from vector_storage import vector_storage


DEFAULT_PROMPT = """\
<|im_start|>system
请简介和专业的回答用户的问题，答案请使用中文。<|im_end|>
<|im_start|>user
问题是：{query}<|im_end|>
<|im_start|>assistant

"""

RAG_PROMPT = """\
<|im_start|>system
已知信息：
```
{context}
```<|im_end|>
<|im_start|>user
根据上述已知信息，简洁和专业的来回答用户的问题，答案请使用中文。
问题是：{query}<|im_end|>
<|im_start|>assistant

"""


def query_llm(query, max_tokens, temperature, use_rag, k):
    """
    根据用户提问，从向量库中检索相关文本，并生成回答
    """
    retrieved_info = ""
    if use_rag:
        vector_search_res = vector_storage.get_similar_documents(query, k)
        context = ""
        for idx, item in enumerate(vector_search_res, start=1):
            context += item["text"] + "\n"
            retrieved_info += (
                f"文档 {idx}:\n文本：{item['text']}\n相似度得分：{item['score']}\n\n"
            )

        prompt_template = PromptTemplate.from_template(RAG_PROMPT)
        prompt = prompt_template.format(context=context, query=query)
    else:
        prompt_template = PromptTemplate.from_template(DEFAULT_PROMPT)
        prompt = prompt_template.format(query=query)

    llm = OpenAI(
        model="internlm2-7b",
        openai_api_base="http://0.0.0.0:23333/v1",
        openai_api_key="EMPTY",
        max_tokens=max_tokens,
        temperature=temperature,
    )
    answer = llm.invoke(prompt)
    return answer


def handle_file_upload(upload_files, chunk_size, chunk_overlap):
    upload_logs = ""
    for upload_file in upload_files:
        if upload_file is not None:
            upload_file_type = upload_file.name.split(".")[-1]
            if upload_file_type in [
                "txt",
                "md",
            ]:
                loader = TextLoader(upload_file)
            elif upload_file_type in ["pdf"]:
                loader = PDFPlumberLoader(upload_file)
            elif upload_file_type in ["docx"]:
                loader = Docx2txtLoader(upload_file)
            else:
                return "上传的文件格式不支持"
            data = loader.load()
            content = data[0].page_content

            # 文档切分
            chunks = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap
            ).split_text(content)

            # 向量存储
            for chunk in chunks:
                vector_storage.add_to_index(chunk)

            log = f"文档 {upload_file.split('/')[-1]}\t 成功上传到知识库\n"
            upload_logs += log

    return upload_logs


if __name__ == "__main__":
    with gr.Blocks() as demo:
        gr.Markdown("# 翰墨政声—公文大语言模型智能问答分析系统")

        with gr.Row():
            with gr.Column():
                user_input = gr.Textbox(
                    label="用户输入",
                    lines=5,
                    max_lines=5,
                )
                submit_btn = gr.Button("提交", variant="primary")
            llm_output = gr.Textbox(
                label="模型输出",
                lines=6,
                max_lines=6,
                interactive=False,
            )

        with gr.Row(variant="compact"):
            with gr.Column():
                gr.Markdown("LLM 设置")
                max_tokens = gr.Slider(
                    256, 32768, value=1024, step=1, label="最大长度", interactive=True
                )
                temperature = gr.Slider(
                    0, 1, value=0.7, step=0.1, label="温度", interactive=True
                )
                with gr.Row():
                    use_rag = gr.Checkbox(label="是否使用 RAG 模型")
                    top_k = gr.Slider(
                        1, 10, value=3, step=1, label="top_k", interactive=True
                    )
            with gr.Column():
                gr.Markdown("知识库管理")
                upload_files = gr.Files(
                    label="请上传知识库文件",
                    file_types=[".txt", ".docx", ".pdf"],
                    interactive=True,
                )
                chunk_size = gr.Slider(
                    100, 1000, value=500, step=100, label="chunk_size", interactive=True
                )
                chunk_overlap = gr.Slider(
                    0, 300, value=100, step=10, label="chunk_overlap", interactive=True
                )
                kb_output = gr.Textbox(
                    label="知识库管理输出日志", lines=5, max_lines=5, interactive=False
                )
                with gr.Row():
                    doc_upload_btn = gr.Button("上传至知识库")
                    empty_btn = gr.Button("清空知识库")

        submit_btn.click(
            fn=query_llm,
            inputs=[user_input, max_tokens, temperature, use_rag, top_k],
            outputs=[llm_output],
        )

        doc_upload_btn.click(
            fn=handle_file_upload,
            show_progress=True,
            inputs=[upload_files, chunk_size, chunk_overlap],
            outputs=[kb_output],
        )

        empty_btn.click(
            fn=vector_storage.clear,
            show_progress=True,
            inputs=[],
            outputs=[kb_output],
        )

    demo.launch(server_name="10.0.65.55")

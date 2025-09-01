import gradio as gr
import os
from typing import List, Tuple

from service.rag_service import RAGService


class GradioApp:
    def __init__(self, rag_service: RAGService):
        self.rag_service = rag_service
        self._build_ui()

    def _build_ui(self):
        with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", secondary_hue="sky"),
                       title="Enterprise RAG System") as self.demo:
            gr.Markdown("# 企业级RAG智能问答系统 (Enterprise RAG System)")
            gr.Markdown("您可以**加载现有知识库**快速开始，或**上传新文档**构建一个全新的知识库。")

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 控制面板 (Control Panel)")

                    self.load_kb_button = gr.Button("加载已有知识库 (Load Existing KB)")

                    gr.Markdown("<hr style='border: 1px solid #ddd; margin: 1rem 0;'>")

                    self.file_uploader = gr.File(
                        label="上传新文档以构建 (Upload New Docs to Build)",
                        file_count="multiple",
                        file_types=[".pdf", ".txt"],
                        interactive=True
                    )
                    self.build_kb_button = gr.Button("构建新知识库 (Build New KB)", variant="primary")

                    self.status_box = gr.Textbox(
                        label="系统状态 (System Status)",
                        value="系统已初始化，等待加载或构建知识库。",
                        interactive=False,
                        lines=4
                    )

                # --- 刚开始隐藏，构建了数据库再显示 ---
                with gr.Column(scale=2, visible=False) as self.chat_area:
                    gr.Markdown("### 对话窗口 (Chat Window)")
                    self.chatbot = gr.Chatbot(label="RAG Chatbot", bubble_full_width=False, height=500)
                    self.mode_selector = gr.Radio(
                        ["流式输出(Streaming)","一次性输出(Full)"],
                        label="输出模式:(Output Mode)",
                        value="流式输出(Streaming)"
                    )
                    self.question_box = gr.Textbox(label="您的问题", placeholder="请在此处输入您的问题...",
                                                   show_label=False)
                    with gr.Row():
                        self.submit_btn = gr.Button("提交 (Submit)", variant="primary")
                        self.clear_btn = gr.Button("清空历史 (Clear History)")

                    gr.Markdown("---")
                    self.source_display = gr.Markdown("### 引用来源 (Sources)")

            # --- Event Listeners ---
            self.load_kb_button.click(
                fn=self._handle_load_kb,
                inputs=None,
                outputs=[self.status_box, self.chat_area]
            )

            self.build_kb_button.click(
                fn=self._handle_build_kb,
                inputs=[self.file_uploader],
                outputs=[self.status_box, self.chat_area]
            )

            self.submit_btn.click(
                fn=self._handle_chat_submission,
                inputs=[self.question_box, self.chatbot, self.mode_selector],
                outputs=[self.chatbot, self.question_box, self.source_display]
            )

            self.question_box.submit(
                fn=self._handle_chat_submission,
                inputs=[self.question_box, self.chatbot, self.mode_selector],
                outputs=[self.chatbot, self.question_box, self.source_display]
            )

            self.clear_btn.click(
                fn=self._clear_chat,
                inputs=None,
                outputs=[self.chatbot, self.question_box, self.source_display]
            )

    def _handle_load_kb(self):
        """处理现有知识库的加载。返回更新字典。"""
        success, message = self.rag_service.load_knowledge_base()
        if success:
            return {
                self.status_box: gr.update(value=message),
                self.chat_area: gr.update(visible=True)
            }
        else:
            return {
                self.status_box: gr.update(value=message),
                self.chat_area: gr.update(visible=False)
            }

    def _handle_build_kb(self, files: List[str], progress=gr.Progress(track_tqdm=True)):
        """构建新知识库，返回更新的字典."""
        if not files:
            # --- MODIFIED LINE ---
            return {
                self.status_box: gr.update(value="错误：请至少上传一个文档。"),
                self.chat_area: gr.update(visible=False)
            }

        file_paths = [file.name for file in files]

        try:
            for status in self.rag_service.build_knowledge_base(file_paths):
                progress(0.5, desc=status)

            final_status = "知识库构建完成并已就绪！√"
            # --- MODIFIED LINE ---
            return {
                self.status_box: gr.update(value=final_status),
                self.chat_area: gr.update(visible=True)
            }
        except Exception as e:
            error_message = f"构建失败: {e}"
            # --- MODIFIED LINE ---
            return {
                self.status_box: gr.update(value=error_message),
                self.chat_area: gr.update(visible=False)
            }

    def _handle_chat_submission(self, question: str, history: List[Tuple[str, str]], mode: str):
        if not question or not question.strip():
            yield history, "", "### 引用来源 (Sources)\n"
            return

        history.append((question, ""))

        try:
            # 一次全部输出
            if "Full" in mode:
                yield history, "", "### 引用来源 (Sources)\n"

                answer, sources = self.rag_service.get_response_full(question)
                # 获取引用内容
                context_string_for_display = self.rag_service.get_context_string(sources)
                # 修改格式
                source_text_for_panel = self._format_sources(sources)
                #完整内容：引用+回答
                full_response = f"{context_string_for_display}\n\n---\n\n**回答 (Answer):**\n{answer}"
                history[-1] = (question, full_response)

                yield history, "", source_text_for_panel

            # 流式输出
            else:
                answer_generator, sources = self.rag_service.get_response_stream(question)

                context_string_for_display = self.rag_service.get_context_string(sources)

                source_text_for_panel = self._format_sources(sources)

                yield history, "", source_text_for_panel

                response_prefix = f"{context_string_for_display}\n\n---\n\n**回答 (Answer):**\n"
                history[-1] = (question, response_prefix)
                yield history, "", source_text_for_panel

                answer_log = ""
                for text_chunk in answer_generator:
                    answer_log += text_chunk
                    history[-1] = (question, response_prefix + answer_log)
                    yield history, "", source_text_for_panel

        except Exception as e:
            error_response = f"处理请求时出错: {e}"
            history[-1] = (question, error_response)
            yield history, "", "### 引用来源 (Sources)\n"

    def _format_sources(self, sources: List) -> str:
        source_text = "### 引用来源 (sources)\n)"
        if not sources:
            return source_text

        unique_sources = set()
        for doc in sources:
            source_name = os.path.basename(doc.metadata.get('source', 'Unknown'))
            page_num = doc.metadata.get('page', 'N/A')
            unique_sources.add(f"- **{source_name}** (Page: {page_num})")

        source_text += "\n".join(sorted(list(unique_sources)))
        return source_text

    def _clear_chat(self):
        """清理聊天内容"""
        return None, "", "### 引用来源 (Sources)\n"

    def launch(self):
        self.demo.queue().launch(server_name="127.0.0.1", server_port=7860, share=True)


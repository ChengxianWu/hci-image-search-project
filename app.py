import gradio as gr
import time
from PIL import Image

from clip_utils import CLIPEncoder
from search_utils import (
    build_result_summary,
    create_results_zip,
    load_index,
    results_to_gallery,
    search_by_vector,
)


print("[INFO] Loading CLIP model...")
encoder = CLIPEncoder()

print("[INFO] Loading image index...")
image_features, image_paths, metadata = load_index()

print("[INFO] System ready.")


def text_search(query_text, top_k):
    if query_text is None or query_text.strip() == "":
        return [], "Please input a text query.", None, "Query preview: Empty"

    query_text = query_text.strip()
    query_vector = encoder.encode_text(query_text)
    results = search_by_vector(query_vector, image_features, image_paths, metadata, top_k=int(top_k))

    return (
        results_to_gallery(results),
        build_result_summary(results),
        create_results_zip(results),
        f"Query preview: {query_text}",
    )


def image_search(query_image, top_k):
    if query_image is None:
        return [], "Please upload an image.", None, "Query preview: No image uploaded"

    if isinstance(query_image, Image.Image):
        image = query_image.convert("RGB")
    else:
        image = Image.open(query_image).convert("RGB")

    query_vector = encoder.encode_image(image)
    results = search_by_vector(query_vector, image_features, image_paths, metadata, top_k=int(top_k))

    return (
        results_to_gallery(results),
        build_result_summary(results),
        create_results_zip(results),
        "Query preview: Uploaded image encoded by CLIP.",
    )


with gr.Blocks(title="HCI Lab 3 - Image Search System") as demo:
    gr.Markdown("# HCI Lab 3: Image Search System")

    with gr.Row():
        gr.Markdown("Formulation")
        gr.Markdown("Preview")
        gr.Markdown("Initiation")
        gr.Markdown("Review")
        gr.Markdown("Refinement")
        gr.Markdown("Use")

    with gr.Tab("Text-to-Image Search"):
        with gr.Row():
            with gr.Column(scale=1):
                text_input = gr.Textbox(
                    label="Text Query",
                    placeholder="apple, milk bottle, red fruit, banana",
                    lines=2,
                )
                text_top_k = gr.Slider(
                    minimum=1,
                    maximum=20,
                    value=5,
                    step=1,
                    label="Number of Returned Images Top-K",
                )
                text_search_button = gr.Button("Search", variant="primary")
                text_query_preview = gr.Textbox(label="Query Preview", interactive=False)

            with gr.Column(scale=2):
                text_summary = gr.Textbox(label="Result Overview", interactive=False)
                text_gallery = gr.Gallery(label="Search Results", columns=5, height=450)
                text_download = gr.File(label="Download Search Results as ZIP")

        text_search_button.click(
            fn=text_search,
            inputs=[text_input, text_top_k],
            outputs=[text_gallery, text_summary, text_download, text_query_preview],
        )

    with gr.Tab("Image-to-Image Search"):
        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(label="Upload Query Image", type="pil")
                image_top_k = gr.Slider(
                    minimum=1,
                    maximum=20,
                    value=5,
                    step=1,
                    label="Number of Returned Images Top-K",
                )
                image_search_button = gr.Button("Search", variant="primary")
                image_query_preview = gr.Textbox(label="Query Preview", interactive=False)

            with gr.Column(scale=2):
                image_summary = gr.Textbox(label="Result Overview", interactive=False)
                image_gallery = gr.Gallery(label="Search Results", columns=5, height=450)
                image_download = gr.File(label="Download Search Results as ZIP")

        image_search_button.click(
            fn=image_search,
            inputs=[image_input, image_top_k],
            outputs=[image_gallery, image_summary, image_download, image_query_preview],
        )


if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(), prevent_thread_lock=True)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        demo.close()

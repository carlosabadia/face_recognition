from model_utils import detect_face
import gradio as gr
import numpy as np

# Function to run the app

def run_model(image: np.ndarray):
    return gr.Image.update(value=detect_face(image))


def interface() -> None:
    """
    Create and launch the graphical user interface face detection app.
    """

    # Create the blocks for the interface
    with gr.Blocks() as app:
        # Add a title and opening HTML element
        gr.HTML(
            """
            <div style="text-align: center; max-width: 650px; margin: 0 auto; padding-top: 7px;">
              <div
                style="
                  display: inline-flex;
                  align-items: center;
                  gap: 0.8rem;
                  font-size: 1.85rem;
                "
              >
                <h1 style="font-weight: 900; margin-bottom: 7px;">
                  Face Detection App 👤
                </h1>
              </div>
            </div>
        """
        )
        with gr.Group():
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        webcam_image_in = gr.Webcam(label="Webcam input")
                    with gr.Row():
                        gr.Text(
                            label="⚠️ Reminder ", value="Do not forget to click the camera button to freeze and get the webcam image 📷!", interactive=False)
                with gr.Column():
                    with gr.Row():
                        face_detected_image_out = gr.Image(
							label="Face detected", interactive=False)
                    with gr.Row():
                        detect_button = gr.Button(value="Detect face 👤")

                    detect_button.click(fn=run_model, inputs=[
						webcam_image_in], outputs=face_detected_image_out)

        app.launch()


if __name__ == '__main__':
    interface()  # Run the interface

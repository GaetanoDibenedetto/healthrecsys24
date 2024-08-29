import fcntl
import json
import os
import random
import time
import torch
import json
import os
import random
import gradio as gr
import torch
import datetime

from mmpose.apis import MMPoseInferencer

from utils.utils import compute_angles, load_keypoint_in_torch, path_model_checkpoint, path_pair_file_angles_stats_json

import transformers

import gradio as gr

from transformers.trainer_utils import set_seed

test = False

global device
device = "cpu"
if torch.cuda.is_available():
    # if GPU available, use cuda (on a cpu, training will take a considerable length of time!)
    device = "cuda"


def predict_single_image(model, result):
    original_keypoint = result["predictions"][0][0]["keypoints"]
    keypoint = load_keypoint_in_torch(original_keypoint)
    keypoint = keypoint.to(device)
    output = model(keypoint)
    return torch.max(output, axis=1)[1], original_keypoint


def inference(img):
    visualization_path = "tmp/vis_results"
    result_generator = mmpose_inferencer(img, show=False, vis_out_dir=visualization_path)
    result = next(result_generator)
    prediction, keypoints = predict_single_image(model_classification, result)

    visualization = os.path.join(visualization_path, img.split("/")[-1])

    if prediction == 0:
        (
            shoulder_level,
            hip_level,
            y_head_mid_torso,
            y_head_upper_torso,
            y_head_bottom_torso,
            x_head_mid_torso,
            x_head_upper_torso,
            x_head_bottom_torso,
        ) = compute_angles(keypoints)

        prompt = "I've an incorrect posture caused by"
        counter = 0
        output_text = "Incorrect Posture"
        if abs(shoulder_level) > angle_stats["avg"]["shoulders"]:
            counter += 1
            prompt += f" uneven shoulders"
            output_text += f"\nUneven shoulders: {shoulder_level}"

        if abs(hip_level) > angle_stats["avg"]["hips"]:
            if counter > 0:
                prompt += " and"
            counter += 1
            prompt += f" uneven hips"
            output_text += f"\nUneven hips: {hip_level}"

        if abs(y_head_mid_torso) > angle_stats["avg"]["y_head_mid_torso"]:
            if counter > 0:
                prompt += " and"
            counter += 1
            prompt += f" forward head posture (Head leaning forward)"
            output_text += (
                f"\nForward Head Posture (Head leaning forward): {y_head_mid_torso}"
            )

        if abs(x_head_mid_torso) > angle_stats["avg"]["x_head_mid_torso"]:
            if counter > 0:
                prompt += " and"
            counter += 1
            prompt += f" head tilted sideways"
            output_text += f"\nHead tilted sideways: {x_head_mid_torso}"

        if counter == 0:
            output_text += f"\nNo uneven body parts detected"
            prompt = "I've an incorrect posture caused which exercise or stretching should I do?"
        else:
            prompt += f", which exercise or stretching should I do?"
        
        if test:
            llama_output = "Test"
        else: llama_output = model_llama.generate_message(prompt)
        
    else:
        output_text = "Correct posture"
        llama_output = None

    return output_text, llama_output, visualization, prediction


def load_model_classification(model_path=None, model_type="keypoint"):
    # load model
    if model_path is None:
        model_path = os.path.join(path_model_checkpoint , model_type)
        
    list_models = os.listdir(model_path)
    list_models.sort()
    model_name = list_models[0]
    model_path = os.path.join(model_path, model_name)
    model = torch.load(model_path)
    model.to(device)
    model.eval()
    return model


def load_keypoint_extractor():
    inferencer = MMPoseInferencer(
        pose3d="mmpose/configs/body_3d_keypoint/motionbert/h36m/"
        "motionbert_dstformer-ft-243frm_8xb32-120e_h36m.py",
        pose3d_weights="https://download.openmmlab.com/mmpose/v1/body_3d_keypoint/"
        "pose_lift/h36m/motionbert_ft_h36m-d80af323_20230531.pth",
        show_progress=False,
        det_model="yolox_l_8x8_300e_coco",
        det_weights="https://download.openmmlab.com/mmdetection/v2.0/"
        "yolox/yolox_l_8x8_300e_coco/"
        "yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth",
        det_cat_ids=[0],  # the category id of 'human' class
    )
    return inferencer


class llama:

    def __init__(self, model_id=None, pipeline=None, max_new_tokens=None):
        if model_id is None:
            self.model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        else:
            self.model_id = model_id

        if pipeline is None:
            self.pipeline = transformers.pipeline(
                "text-generation",
                model=self.model_id,
                model_kwargs={"torch_dtype": torch.bfloat16},
                device_map="auto",
            )
        else:
            self.pipeline = pipeline

        if max_new_tokens is None:
            self.max_new_tokens = 1024
        else:
            self.max_new_tokens = max_new_tokens

    def generate_message(self, text):
        input = [
            {
                "role": "system",
                "content": "You are a medical assistant able to suggest a specific physical exercise for users with uneven body parts to prevent an increase in their unease.",
            },
            {"role": "user", "content": f"{text}"},
        ]
        output = self.pipeline(
            input,
            max_new_tokens=self.max_new_tokens,
        )
        return output[0]["generated_text"][-1]["content"]


global model_classification
model_classification = load_model_classification()
global mmpose_inferencer
mmpose_inferencer = load_keypoint_extractor()
global model_llama
if test:
    model_llama = None
else:
    model_llama = llama()
global angle_stats

with open(path_pair_file_angles_stats_json) as f:
    angle_stats = json.load(f)

DEFAULT_WELCOME_MESSAGE = "Welcome! Please upload or chose an image to get started."

RATE_LABELS = [
f"1. Do you agree with the recommended exercises based on the user’s posture analysis?\n Rating: 1 (Totally Disagree) to 5 (Totally Agree)",

f"2. Do the recommended exercises adequately address the user’s posture issue and imbalances?\n Rating: 1 (Totally Inadequate) to 5 (Totally Adequate)",

f"3. Are the exercises clearly explained and easy to understand?\n Rating: 1 (Very Unclear) to 5 (Very Clear)",

f"4. In your opinion, how effective would these exercises be in improving the user's posture?\n Rating: 1 (Not Effective at All) to 5 (Highly Effective)",

f"5. Are the recommended exercises simple enough for the user to perform without expert supervision?\n Rating: 1 (Not Simple at All) to 5 (Very Simple)",
]

frames_path = os.path.join("archives_data", "frames")
if len(os.listdir(frames_path)) < 20:
    examples_frames = os.listdir(frames_path)
else:
    examples_frames = random.sample(os.listdir(frames_path), 20)
# examples_frames = os.listdir(frames_path)
for i, frame in enumerate(examples_frames):
    examples_frames[i] = [os.path.join(frames_path, frame)]


with gr.Blocks(
    title="Web App", fill_height=True, theme=gr.themes.Default(text_size="lg")
) as demo:

    with gr.Column(show_progress=False):
        chatbot = gr.Chatbot(
            value=[(None, DEFAULT_WELCOME_MESSAGE)], scale=1, label=''
        )

    with gr.Column():
        with gr.Row():
            msg = gr.Textbox(
                lines=1,
                label="What improvements would you suggest for this recommendation?",
                placeholder="In italiano va anche bene!",
                interactive=True,
                scale=1,
                visible=False,
            )

    with gr.Column():
        with gr.Row():
            rate_story = gr.Radio([1, 2, 3, 4, 5], visible=False)

    with gr.Column() as input_box:
        with gr.Row():
            submit_button = gr.Button("Submit")

        with gr.Row():
            image_input = gr.Image(label="Input Image", type="filepath")

        with gr.Row():
            list_examples = gr.Examples(examples=examples_frames, inputs=image_input, examples_per_page=20)    

    with gr.Column(visible=False) as restart_box:
        with gr.Row():
            restart_completed = gr.Label(
                "Done! Thank you for your time!", 
            )
        with gr.Row():
            restart_button = gr.Button("Restart", )
            restart_clear_chat_button = gr.Button(
                "Restart and Clear Chat", 
            )

    def restart():
        return (
            gr.update(value=None, visible=False),
            gr.update(value=None, visible=False),
            gr.update(value=None, visible=True),
            gr.update(visible=True),
            gr.update(visible=False),
        )

    def restart_clear_chat():
        return (
            gr.update(value=None, visible=False),
            gr.update(value=None, visible=False),
            gr.update(value=None, visible=True),
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(value=[(None, DEFAULT_WELCOME_MESSAGE)]),
        )

    restart_button.click(restart, outputs=[msg, rate_story, image_input, input_box, restart_box])
    restart_clear_chat_button.click(
        restart_clear_chat, outputs=[msg, rate_story, image_input, input_box, restart_box, chatbot]
    )

    def respond(image_path, chat_history, rate_story):

        set_seed(42)
        detection, llama_output, vis_motion_bert, prediction = inference(image_path)
        chat_history.append(("Input image:", gr.Image(image_path)))
        chat_history.append(("MotionBert:", gr.Image(vis_motion_bert)))
        chat_history.append(("Detection:", detection))
        if prediction == 0:
            chat_history.append(("Reccomendation:", llama_output))
            return (
                chat_history,
                gr.update(visible=False),
                gr.update(visible=True, label=RATE_LABELS[0]),
                gr.update(value=None, visible=False),
                gr.update(visible=False),
            )

        return (
            chat_history,
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(value=None, visible=False),
            gr.update(visible=True),
        )

    submit_button.click(
        respond,
        [image_input, chatbot, rate_story],
        [chatbot, msg, rate_story, input_box, restart_box],
    )

    def rate(choice, chat_history, restart):

        if choice is not None:

            counter = 0

            choice_str = f"Rating {choice} was received and registered!"
            chat_history.append((None, choice_str))
            time.sleep(2)

            chat_history_len = len(chat_history)

            for idx in range(chat_history_len):
                actual_idx = chat_history_len - idx - 1

                if chat_history[actual_idx][0] is None:
                    counter += 1
                else:
                    break

            if counter == len(RATE_LABELS):
                return (
                    gr.update(value=None, visible=False),
                    chat_history,
                    gr.update(visible=True),
                )

            else:
                return (
                    gr.update(value=None, label=RATE_LABELS[counter]),
                    chat_history,
                    gr.update(visible=False),
                )

        return choice, chat_history, gr.update()

    rate_story.change(
        fn=rate,
        inputs=[rate_story, chatbot, restart_button],
        outputs=[rate_story, chatbot, msg],
    )

    def expert_answer(text, chat_history, rate_story):
        chat_history.append((None, text))

        with open("tmp/conversations.jsonl", "a", encoding="utf8") as f:

            fcntl.flock(f, fcntl.LOCK_EX)

            complete_dialogue = chat_history[
                len(chat_history) - (len(RATE_LABELS) + 3) :
            ]
            dialogue_to_save = []

            for user_msg, sys_msg in complete_dialogue:
                dialogue_to_save.append({"user": user_msg, "system": sys_msg})

            json.dump({"dialogue": dialogue_to_save}, f)
            f.write(f"\n{datetime.datetime.now()}\n\n")

            fcntl.flock(f, fcntl.LOCK_UN)
            chat_history.append((None, "Survey completed!"))   

        return  gr.update(value=None, visible=False), chat_history, gr.update(visible=True)

    msg.submit(expert_answer, [msg, chatbot, rate_story], [msg, chatbot, restart_box])


demo.launch(share=True)

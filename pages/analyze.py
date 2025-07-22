
"""
Main page of the app
"""

import json
import requests
import threading

import flet as ft


url = "http://localhost:8000/analyze"  # TODO: it is hardcoded for now here, later take it from config or smth


def send_image(image_path: str):
    try:
        with open(image_path, "rb") as f:
            files = {"file": (image_path.split("/")[-1], f, "image/jpeg")}
            response = requests.post(url, files=files)
            if response.status_code == 200:
                return response.json()  # return JSON response here
            else:
                return {"status": response.status_code}
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return {"status": 500}


def analyze_view(page: ft.Page):

    def go_to_upload(e):
        loading.visible = False
        info_controls.visible = False
        upload_controls.visible = True
        content_column.visible = False
        page.update()
    def close_error(e):
        if error_controls.visible:
            error_controls.visible = False
            page.update()

    error_text = ft.Text("", size=12, weight=ft.FontWeight.BOLD, color="red")
    close_button = ft.IconButton(
        icon=ft.Icons.CLOSE,
        tooltip="Close",
        on_click=close_error,
        icon_color=ft.Colors.RED
    )
    error_controls = ft.Row(controls=[error_text, close_button], alignment=ft.MainAxisAlignment.CENTER,
                             visible=False)

    upload_controls = ft.Column(
        controls=[
            ft.Text("Upload Image", size=24, weight=ft.FontWeight.BOLD),
            ft.ElevatedButton(
                "Pick Image",
                on_click=lambda e: file_picker.pick_files(
                    file_type=ft.FilePickerFileType.IMAGE
                )
            ),
        ],
        alignment=ft.MainAxisAlignment.CENTER,
        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
    )
    loading = ft.ProgressRing()
    loading.value = 'Analyzing image'

    # Plant title text (large font)
    plant_title = ft.Text("", size=30, weight=ft.FontWeight.BOLD)
    back = ft.TextButton("Go Back", on_click=go_to_upload)

    # Image display
    image_display = ft.Image(width=400, height=400, fit=ft.ImageFit.CONTAIN)

    # Characteristics table (below image)
    characteristics_column = ft.Column(spacing=5, horizontal_alignment=ft.CrossAxisAlignment.CENTER, width=200)
    summary_text = ft.Text(
        value="",
        size=14,
        italic=True,
        text_align=ft.TextAlign.CENTER,
        max_lines=5,
        overflow=ft.TextOverflow.ELLIPSIS,
        # soft_wrap=True,
        width=400,  # Same width as column or slightly less
    )

    # Main content column with all elements except loading
    content_column = ft.Column(
        controls=[back, plant_title, image_display, characteristics_column, summary_text],
        visible=False,
        alignment=ft.MainAxisAlignment.CENTER,
        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        spacing=15,
    )
    info_controls = ft.Column([loading, content_column], visible=False)


    def pick_file_result(e):
        data_dict = json.loads(e.data)

        error_text.value = ''
        error_controls.visible = False

        if data_dict.get("files"):
            file_path = data_dict["files"][0]["path"]
            print(f"File path: {file_path}")

            loading.visible = True
            info_controls.visible = True
            upload_controls.visible = False
            content_column.visible = False
            page.update()  # Allows UI to show loading

            # Handle request in separate thread
            def process_file():
                try:
                    result = send_image(file_path)
                except Exception as ex:
                    error_text = f'Error: {ex}'
                    error_controls.visible = True
                    upload_controls.visible = True
                    info_controls.visible = False
                    content_column.visible = False
                    loading.visible = False

                loading.visible = False

                if isinstance(result, dict) and "data" in result and "image_base64" in result:
                    # Update the image and details
                    image_display.src_base64 = result["image_base64"]
                    plant_data = result["data"]
                    plant_title.value = plant_data.get("Plant", "Unknown Plant")

                    # Clear and populate the characteristics column
                    characteristics_column.controls.clear()
                    for key, val in plant_data.items():
                        if key != "Plant" and key != "Summary":
                            characteristics_column.controls.append(
                                ft.Row([
                                    ft.Text(f"{key}:", weight=ft.FontWeight.BOLD, width=100),
                                    ft.Text(str(val)),
                                ])
                            )

                    summary_text.value = plant_data.get("Summary", "")

                    content_column.visible = True
                    info_controls.visible = True
                    error_controls.visible = False
                    upload_controls.visible = False

                else:
                    error_text.value = f"Error: Server response was {result.get('status', 'Unknown')}"
                    error_controls.visible = True
                    upload_controls.visible = True
                    info_controls.visible = False

                page.update()

            threading.Thread(target=process_file).start()
        else:
            error_text.value = "Error: Path cannot be found!"
            error_controls.visible = True
            upload_controls.visible = True
            info_controls.visible = False
            loading.visible = False
            page.update()

    file_picker = ft.FilePicker(on_result=pick_file_result)
    page.overlay.append(file_picker)

    return ft.View(
        route="/analyze",
        vertical_alignment=ft.MainAxisAlignment.CENTER,  # vertical centering
        horizontal_alignment=ft.CrossAxisAlignment.CENTER,  # horizontal centering
        controls=[
            error_controls,
            upload_controls,
            info_controls
        ]
    )
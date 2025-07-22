import json
import requests
import threading
import flet as ft

url = "http://localhost:8000/analyze"  # TODO: config later


def send_image(image_path: str):
    """Send image file to the server and return JSON response."""
    try:
        with open(image_path, "rb") as f:
            files = {"file": (image_path.split("/")[-1], f, "image/jpeg")}
            response = requests.post(url, files=files)
            if response.status_code == 200:
                return response.json()
            else:
                return {"status": response.status_code}
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return {"status": 500}


def create_error_controls(on_close):
    error_text = ft.Text("", size=12, weight=ft.FontWeight.BOLD, color="red")
    close_button = ft.IconButton(
        icon=ft.Icons.CLOSE,
        tooltip="Close",
        on_click=on_close,
        icon_color=ft.Colors.RED,
    )
    error_controls = ft.Row(
        controls=[error_text, close_button],
        alignment=ft.MainAxisAlignment.CENTER,
        visible=False,
    )
    return error_controls, error_text


def create_upload_controls(on_pick_image):
    upload_controls = ft.Column(
        controls=[
            ft.Text("Upload Image", size=24, weight=ft.FontWeight.BOLD),
            ft.ElevatedButton("Pick Image", on_click=on_pick_image),
        ],
        alignment=ft.MainAxisAlignment.CENTER,
        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
    )
    return upload_controls


def analyze_view(page: ft.Page):

    # Callback handlers
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

    def on_pick_image(e):
        file_picker.pick_files(file_type=ft.FilePickerFileType.IMAGE)

    # Create controls
    error_controls, error_text = create_error_controls(close_error)
    upload_controls = create_upload_controls(on_pick_image)
    loading = ft.ProgressRing(visible=False)

    plant_title = ft.Text("", size=30, weight=ft.FontWeight.BOLD)
    back = ft.TextButton("Go Back", on_click=go_to_upload)
    image_display = ft.Image(width=400, height=400, fit=ft.ImageFit.CONTAIN)
    characteristics_column = ft.Column(spacing=5, horizontal_alignment=ft.CrossAxisAlignment.CENTER, width=200)
    summary_text = ft.Text(
        value="",
        size=14,
        italic=True,
        text_align=ft.TextAlign.CENTER,
        max_lines=20,
        overflow=ft.TextOverflow.ELLIPSIS,
        width=400,
    )

    content_column = ft.Column(
        controls=[back, plant_title, image_display, characteristics_column, summary_text],
        visible=False,
        alignment=ft.MainAxisAlignment.CENTER,
        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        spacing=15,
    )

    info_controls = ft.Container(
        content=ft.Column(
            controls=[loading, content_column],
            alignment=ft.MainAxisAlignment.CENTER,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=20,
        ),
        expand=True,
    )

    # File picker with handler
    def pick_file_result(e):
        data_dict = json.loads(e.data)
        error_text.value = ""
        error_controls.visible = False

        if data_dict.get("files"):
            file_path = data_dict["files"][0]["path"]
            print(f"File path: {file_path}")

            loading.visible = True
            info_controls.visible = True
            upload_controls.visible = False
            content_column.visible = False
            page.update()

            def process_file():
                try:
                    result = send_image(file_path)
                except Exception as ex:
                    error_text.value = f"Error: {ex}"
                    error_controls.visible = True
                    upload_controls.visible = True
                    info_controls.visible = False
                    content_column.visible = False
                    loading.visible = False
                    page.update()
                    return

                loading.visible = False

                if isinstance(result, dict) and "data" in result and "image_base64" in result:
                    image_display.src_base64 = result["image_base64"]
                    plant_data = result["data"]
                    plant_title.value = plant_data.get("Plant", "Unknown Plant")

                    characteristics_column.controls.clear()
                    for key, val in plant_data.items():
                        if key not in ("Plant", "Summary"):
                            characteristics_column.controls.append(
                                ft.Row(
                                    [
                                        ft.Text(f"{key}:", weight=ft.FontWeight.BOLD, width=100),
                                        ft.Text(str(val)),
                                    ]
                                )
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
        controls=[
            ft.Container(
                expand=True,
                alignment=ft.alignment.center,
                content=ft.Column(
                    expand=True,
                    alignment=ft.MainAxisAlignment.CENTER,
                    horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                    scroll=ft.ScrollMode.AUTO,
                    controls=[
                        error_controls,
                        upload_controls,
                        info_controls,
                    ],
                ),
            )
        ],
    )

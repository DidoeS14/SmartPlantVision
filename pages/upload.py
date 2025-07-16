"""
Main page of the app
"""

import json
import requests

import flet as ft


url = "http://localhost:8000/upload" # TODO: it is hardcoded for now here, later take it from config or smth


def send_image(image_path: str):
    with open(image_path, "rb") as f:
        files = {"file": (image_path.split("/")[-1], f, "image/jpeg")}  # or "image/png"
        response = requests.post(url, files=files)

    # Print server response
    print("Status code:", response.status_code)
    print("Response:", response.text)
    return response.status_code


def upload_view(page: ft.Page):
    image_display = ft.Column([])  # Container to hold image once it's selected

    def pick_file_result(e):
        data_dict = json.loads(e.data)

        # Now access the file path
        if data_dict.get("files"):
            file_path = data_dict["files"][0]["path"]
            print(f"File path: {file_path}")
            response = send_image(file_path)
            if response == 200:
                page.go("/info")
        else:
            print('Path cannot be found!')

    file_picker = ft.FilePicker(on_result=pick_file_result)
    page.overlay.append(file_picker)

    return ft.View(
        route="/upload",
        vertical_alignment=ft.MainAxisAlignment.CENTER,  # vertical centering
        horizontal_alignment=ft.CrossAxisAlignment.CENTER,  # horizontal centering
        controls=[
            ft.Column(
                controls=[
                    ft.Text("Upload Image", size=24, weight=ft.FontWeight.BOLD),
                    ft.ElevatedButton(
                        "Pick Image",
                        on_click=lambda e: file_picker.pick_files(
                            file_type=ft.FilePickerFileType.IMAGE
                        )
                    ),
                    image_display
                ],
                alignment=ft.MainAxisAlignment.CENTER,
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            )
        ]
    )
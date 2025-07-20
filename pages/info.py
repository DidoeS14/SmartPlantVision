import flet as ft
import threading
import requests
import asyncio

data_store = {"image_base64": None, "ready": False}

def fetch_data_background():
    try:
        response = requests.get("http://localhost:8000/info")
        if response.status_code == 200:
            data = response.json()
            data_store["image_base64"] = data.get("image_base64", "")
            data_store["data"] = data.get("data", {})  # <-- Save the dictionary with Status, Plant, etc.
        else:

            data_store["data"] = {}
    except Exception as e:
        data_store["data"] = {}
    data_store["ready"] = True

def info_view(page: ft.Page):
    loading = ft.ProgressRing()
    back = ft.TextButton("Go Back", on_click=lambda e: page.go("/upload"))

    # Plant title text (large font)
    plant_title = ft.Text("", size=30, weight=ft.FontWeight.BOLD)

    # Image display
    image_display = ft.Image(width=400, height=400, fit=ft.ImageFit.CONTAIN)

    # Characteristics table (below image)
    characteristics_column = ft.Column(spacing=5, horizontal_alignment=ft.CrossAxisAlignment.CENTER, width=200)

    # Main content column with all elements except loading
    content_column = ft.Column(
        controls=[plant_title, image_display, characteristics_column, back],
        visible=False,
        alignment=ft.MainAxisAlignment.CENTER,
        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        spacing=15,
    )

    page.controls.clear()
    page.controls.append(ft.Column([loading, content_column]))
    page.update()

    async def poll_data():
        for _ in range(40):  # poll max 20 seconds (40 * 0.5s)
            await asyncio.sleep(0.5)
            if data_store["ready"]:
                # Update UI elements
                if data_store["image_base64"]:
                    image_display.src_base64 = data_store["image_base64"]

                # Set Plant title (persistent above image)
                plant = data_store["data"].get("Plant", "Unknown Plant")
                plant_title.value = plant

                # Clear previous characteristics
                characteristics_column.controls.clear()

                # Add all other characteristics below image (excluding Plant)
                for key, val in data_store["data"].items():
                    if key != "Plant":
                        characteristics_column.controls.append(
                            ft.Row([
                                ft.Text(f"{key}:", weight=ft.FontWeight.BOLD, width=100),
                                ft.Text(str(val)),
                            ])
                        )

                loading.visible = False
                content_column.visible = True

                page.update()
                break

    threading.Thread(target=fetch_data_background, daemon=True).start()
    asyncio.create_task(poll_data())

    return ft.View(
        route="/info",
        controls=page.controls,
        vertical_alignment=ft.MainAxisAlignment.CENTER,
        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
    )

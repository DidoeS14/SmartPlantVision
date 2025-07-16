import flet as ft
import threading
import requests
import asyncio

data_store = {"text": "", "image_base64": None, "ready": False}

def fetch_data_background():
    try:
        response = requests.get("http://localhost:8000/info")
        if response.status_code == 200:
            data = response.json()
            data_store["text"] = data.get("text", "No text provided")
            data_store["image_base64"] = data.get("image_base64", "")
        else:
            data_store["text"] = f"Failed with status {response.status_code}"
    except Exception as e:
        data_store["text"] = f"Error: {e}"
    data_store["ready"] = True

async def poll_data(page: ft.Page, info_text: ft.Text, loading: ft.ProgressRing, content_column: ft.Column, image_display: ft.Image):
    for _ in range(40):  # poll max 20 seconds (40 * 0.5s)
        await asyncio.sleep(0.5)
        if data_store["ready"]:
            # Update UI elements
            info_text.value = data_store["text"]
            if data_store["image_base64"]:
                # print(data_store['image_base64'])
                image_display.src_base64 = data_store['image_base64']
            loading.visible = False
            content_column.visible = True

            # Must update page after modifying controls
            page.update()
            break

def info_view(page: ft.Page):
    loading = ft.ProgressRing()
    info_text = ft.Text("Loading...")
    image_display = ft.Image(width=400, height=400, fit=ft.ImageFit.CONTAIN)
    content_column = ft.Column([info_text, image_display], visible=False)

    page.controls.clear()
    page.controls.append(ft.Column([loading, content_column]))
    page.update()

    threading.Thread(target=fetch_data_background, daemon=True).start()
    asyncio.create_task(poll_data(page, info_text, loading, content_column, image_display))

    return ft.View(route="/info", controls=page.controls)
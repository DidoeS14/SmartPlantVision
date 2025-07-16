import flet as ft

from pages.info import info_view

def main(page: ft.Page):
    page.title = "My Android App"

    # Route change handler must be defined before assigning it
    async def route_change(e):
        page.views.clear()

        if page.route == "/info":
            view = info_view(page)  # await here!
            page.views.append(view)
        else:
            page.views.append(ft.View(route=page.route, controls=[ft.Text("Page not found")]))
        page.update()

    page.on_route_change = route_change
    page.go("/info")

ft.app(target=main, view=ft.AppView.FLET_APP)

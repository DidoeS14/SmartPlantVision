import flet as ft

from pages.login import login_view
from pages.register import register_view
from pages.upload import upload_view
from pages.info import info_view

def main(page: ft.Page):
    page.title = "Smart GV"

    page.theme = ft.Theme(
        color_scheme=ft.ColorScheme(
            primary=ft.Colors.GREEN,
            primary_container=ft.Colors.GREEN_200,# TODO: play around and choose nice scheme
            # ...
        ),
    )

    # Route change handler must be defined before assigning it
    async def route_change(e):
        page.views.clear()
        if page.route == "/login":
            page.views.append(login_view(page))
        elif page.route == "/register":
            page.views.append(register_view(page))
        elif page.route == "/upload":
            page.views.append(upload_view(page))
        elif page.route == "/info":
            page.views.append(info_view(page))
        else:
            page.views.append(ft.View(route=page.route, controls=[ft.Text("Page not found")]))
        page.update()

    page.on_route_change = route_change
    page.go("/login")

ft.app(target=main, view=ft.AppView.FLET_APP)

import flet as ft

from pages.login import login_view
from pages.register import register_view
# from pages.upload import upload_view
# from pages.info import info_view
from pages.analyze import analyze_view


# TODO: auth memory token in future
def main(page: ft.Page):
    page.title = "Smart PV"

    page.theme = ft.Theme(
        color_scheme=ft.ColorScheme(
            primary=ft.Colors.GREEN_400,  # Main plant green
            primary_container=ft.Colors.GREEN_900,  # Deep green background container
            secondary=ft.Colors.LIGHT_GREEN_400,  # Light green accents
            secondary_container=ft.Colors.GREEN_800,  # Secondary container
            surface=ft.Colors.BLACK,  # Surface background
            background=ft.Colors.BLACK,  # App background
            error=ft.Colors.RED_400,  # Error red
            on_primary=ft.Colors.BLACK,  # Text on primary buttons
            on_secondary=ft.Colors.BLACK,  # Text on secondary
            on_surface=ft.Colors.WHITE,  # Regular text
            on_background=ft.Colors.WHITE,  # Text on background
            on_error=ft.Colors.BLACK,
            # brightness=ft.Brightness.DARK,  # Dark theme brightness
        )
    )
    # page.theme = ft.Theme(color_scheme_seed=ft.Colors.GREEN)


    # TODO: clean up the pages code
    # Route change handler must be defined before assigning it
    async def route_change(e):
        page.views.clear()
        if page.route == "/login":
            page.views.append(login_view(page))
        elif page.route == "/register":
            page.views.append(register_view(page))
        # elif page.route == "/upload":
        #     page.views.append(upload_view(page))
        # elif page.route == "/info":
        #     page.views.append(info_view(page))
        elif page.route == "/analyze":
            page.views.append(analyze_view(page))
        else:
            page.views.append(ft.View(route=page.route, controls=[ft.Text("Page not found")]))
        page.update()

    page.on_route_change = route_change
    page.go("/login")  # if not logged in, otherwise go to analyse


ft.app(target=main, view=ft.AppView.FLET_APP, assets_dir='assets', )

import pyrebase

import flet as ft
import sensitive    # contains sensitive data, for that reason it is not included in the project


class Debug:
    """
    Used for setting debug variables
    """
    no_auth = True


class URLs:
    """For storing different urls"""

    base = 'http://localhost:8000'
    analyze = '/analyze'


class Firebase:
    """Used for working with firebase"""
    firebaseConfig = sensitive.firebaseConfig   # contains config data with api for firebase (not included in project)

    app = pyrebase.initialize_app(firebaseConfig)
    auth = app.auth()


class StandardControls:
    """Used for creating standardized controls across the app"""

    @staticmethod
    def create_error_controls(page):
        error_text = ft.Text("", size=12, weight=ft.FontWeight.BOLD, color="red")

        def close_error(e):
            # error_text.visible = False
            error_controls.visible = False
            page.update()

        close_button = ft.IconButton(
            icon=ft.Icons.CLOSE,
            tooltip="Close",
            on_click=close_error,
            icon_color=ft.Colors.RED,
        )

        error_controls = ft.Row(
            controls=[error_text, close_button],
            alignment=ft.MainAxisAlignment.CENTER,
            visible=False,
        )

        return error_controls, error_text  # error text is returned also for better control

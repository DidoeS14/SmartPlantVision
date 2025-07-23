import pyrebase

import flet as ft


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
    firebaseConfig = {
        'apiKey': "AIzaSyBjZ7e0nMHixZ3E-FtoNG42SS1RyI0m2SM",
        'authDomain': "smart-plant-vision.firebaseapp.com",
        'projectId': "smart-plant-vision",
        'storageBucket': "smart-plant-vision.firebasestorage.app",
        'messagingSenderId': "173088318294",
        'appId': "1:173088318294:web:84e07f546c760f34862cf7",
        'measurementId': "G-9E9SV36RH7",
        'databaseURL': "https://smart-plant-vision-default-rtdb.firebaseio.com/"
    }

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

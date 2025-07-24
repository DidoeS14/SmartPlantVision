import pyrebase

import flet as ft
import sensitive    # contains sensitive data, for that reason it is not included in the project



class Config:
    """Different non user controllable configurations"""
    unnaceptable_accuracy = 85
    minimum_confident_accuracy = 90
    high_accuracy = 95

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

    @staticmethod
    def create_warning_controls(page):
        text = ft.Text("", size=12, weight=ft.FontWeight.BOLD, color="orange")

        def close_error(e):
            # error_text.visible = False
            controls.visible = False
            page.update()

        close_button = ft.IconButton(
            icon=ft.Icons.CLOSE,
            tooltip="Close",
            on_click=close_error,
            icon_color=ft.Colors.RED,
        )

        controls = ft.Row(
            controls=[text, close_button],
            alignment=ft.MainAxisAlignment.CENTER,
            visible=False,
        )

        return controls, text  # text is returned also for better control

    @staticmethod
    def create_logo_title(has_text: bool = False):
        """
        Creates a logo image element for the given page
        :param has_text: Creates the version without the text
        :return:
        """
        source = 'assets/title.png' if has_text else 'assets/logo.png'
        image = ft.Image(src=source, width=250)
        row = ft.Row(
            controls=[image],
            alignment=ft.MainAxisAlignment.CENTER,
        )
        return row

    @staticmethod
    def create_popup(page, title: str, content: str):
        """
        Creates a customizable popup aler. It has to be assigned to page.dialog and be displayed in the page controls
        :param page: page you are working with
        :param title: Title of the popup
        :param content: Text inside of the popup
        :return:
        """

        # Function to close the popup
        def close_popup(e):
            dialog.open = False
            page.update()

        dialog = ft.AlertDialog(
            title=ft.Text(title),
            content=ft.Text(content),
            actions=[
                ft.TextButton("OK", on_click=close_popup)
            ],
            modal=True
        )

        return dialog

"""
Registration page of the app
"""

import flet as ft
from public import Firebase, StandardControls


def register_view(page: ft.Page):
    text = ft.Text("Register", size=24, weight="bold")
    name = ft.TextField(label="Name", width=300)
    family = ft.TextField(label="Family Name", width=300)
    email = ft.TextField(label="Email", width=300)
    password = ft.TextField(label="Password", password=True, width=300)
    reg_btn = ft.ElevatedButton("Register", width=300)
    back_btn = ft.TextButton("Back to Login", on_click=lambda e: page.go("/login"))

    error, error_text = StandardControls.create_error_controls(page)

    page.update()

    def on_register(e):
        # Handle registration
        user = None

        if not name.value or not family.value:
            error_text.value = 'No value for name or family name!'
            error.visible = True
            page.update()
            return

        if not email.value or not password.value:
            error_text.value = 'No value for email or password!'
            error.visible = True
            page.update()
            return

        try:
            user = Firebase.auth.create_user_with_email_and_password(
                email=email.value,
                password=password.value
            )
            page.go("/login")

        except:
            error_text.value = 'This account already exists!'
            error.visible = True
            page.update()

    reg_btn.on_click = on_register

    return ft.View(
        route="/register",
        controls=[
            ft.Container(
                content=ft.Column(
                    [
                        text,
                        error,
                        name,
                        family,
                        email,
                        password,
                        reg_btn,
                        back_btn
                    ],
                    horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                    spacing=20,
                ),
                alignment=ft.alignment.center,
                expand=True,
            )
        ],
    )

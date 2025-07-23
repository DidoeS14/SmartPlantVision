"""
Login page of the app
"""

import flet as ft
from public import Firebase, StandardControls, Debug


def login_view(page: ft.Page):
    error, error_text = StandardControls.create_error_controls(page)
    title = ft.Text("Login", size=24, weight="bold")
    email = ft.TextField(label="Email", width=300)
    password = ft.TextField(label="Password", password=True, width=300)
    login_btn = ft.ElevatedButton("Login", width=300)
    reg_btn = ft.TextButton("Go to Register", on_click=lambda e: page.go("/register"))

    def on_login(e):

        if Debug.no_auth:
            page.go('/analyze')

        if not email.value or not password.value:
            error_text.value = 'No value for email or password!'
            error.visible = True
            page.update()
            return

        try:
            user = Firebase.auth.sign_in_with_email_and_password(email=email.value, password=password.value)
            page.go('/analyze')

        except:
            error_text.value = 'Incorrect email or password!'
            error.visible = True
            page.update()

    login_btn.on_click = on_login

    return ft.View(
        route="/login",
        controls=[
            ft.Container(
                content=ft.Column(
                    [
                        title,
                        error,
                        email,
                        password,
                        login_btn,
                        reg_btn,
                    ],
                    horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                    spacing=20,
                ),
                alignment=ft.alignment.center,
                expand=True,
            )
        ],
    )

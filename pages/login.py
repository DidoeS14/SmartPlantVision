"""
Login page of the app
"""

import flet as ft

def login_view(page: ft.Page):
    def on_login(e):
        # Validate credentials (mock)
        page.go("/upload")

    return ft.View(
        route="/login",
        controls=[
            ft.Text("Login"),
            ft.TextField(label="Username"),
            ft.TextField(label="Password", password=True),
            ft.ElevatedButton("Login", on_click=on_login),
            ft.TextButton("Go to Register", on_click=lambda e: page.go("/register"))
        ],
    )

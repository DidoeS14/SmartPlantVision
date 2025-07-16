"""
Registration page of the app
"""

import flet as ft

def register_view(page: ft.Page):
    def on_register(e):
        # Handle registration
        page.go("/login")

    return ft.View(
        route="/register",
        controls=[
            ft.Text("Register"),
            ft.TextField(label="Username"),
            ft.TextField(label="Password", password=True),
            ft.ElevatedButton("Register", on_click=on_register),
            ft.TextButton("Back to Login", on_click=lambda e: page.go("/login"))
        ],
    )

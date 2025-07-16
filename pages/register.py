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
            ft.Container(
                content=ft.Column(
                    [
                        ft.Text("Register", size=24, weight="bold"),
                        ft.TextField(label="Name", width=300),
                        ft.TextField(label="Family Name", width=300),
                        ft.TextField(label="Email", width=300),
                        ft.TextField(label="Password", password=True, width=300),
                        ft.ElevatedButton("Register", on_click=on_register, width=300),
                        ft.TextButton("Back to Login", on_click=lambda e: page.go("/login")),
                    ],
                    horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                    spacing=20,
                ),
                alignment=ft.alignment.center,
                expand=True,
            )
        ],
    )

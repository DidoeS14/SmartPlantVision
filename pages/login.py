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
            ft.Container(
                content=ft.Column(
                    [# TODO get logo from server. The server should serve it from a contnets page
                        ft.Text("Login", size=24, weight="bold"),
                        ft.TextField(label="Email", width=300),
                        ft.TextField(label="Password", password=True, width=300),
                        ft.ElevatedButton("Login", on_click=on_login, width=300),
                        ft.TextButton("Go to Register", on_click=lambda e: page.go("/register")),
                    ],
                    horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                    spacing=20,
                ),
                alignment=ft.alignment.center,
                expand=True,
            )
        ],
    )

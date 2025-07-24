# import flet as ft
#
# from pages.info import info_view
#
# def main(page: ft.Page):
#     page.title = "My Android App"
#
#     # Route change handler must be defined before assigning it
#     async def route_change(e):
#         page.views.clear()
#
#         if page.route == "/info":
#             view = info_view(page)  # await here!
#             page.views.append(view)
#         else:
#             page.views.append(ft.View(route=page.route, controls=[ft.Text("Page not found")]))
#         page.update()
#
#     page.on_route_change = route_change
#     page.go("/info")
#
# ft.app(target=main, view=ft.AppView.FLET_APP)


import requests

def get_wikipedia_summary(query):
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{query.replace(' ', '_')}"
    # print(f"Fetching: {url}")  # Debug URL
    response = requests.get(url)
    # print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        # print(data)  # Debug full response
        return data.get("extract")
    return None

print(get_wikipedia_summary("Tomato "))




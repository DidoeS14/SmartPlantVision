
![Logo](https://raw.githubusercontent.com/DidoeS14/SmartPlantVision/main/src/assets/title.png)


# 🌿 Smart Plant Vision

**Smart PV** is a mobile-friendly image analysis app built using [Flet](https://flet.dev/), designed to run seamlessly on Android and iOS. It allows users to upload plant images for automated analysis, classification, and educational feedback.

---

## ✨ Features

- 📱 **Mobile-first UI** using Flet framework
- 🌱 **Plant image recognition** powered by a backend model
- 🔐 **Firebase authentication** for secure login/register
- 🌐 **Wikipedia integration** for fun and informative plant facts
- 💾 **Smart data collection** — some uploaded images are collected (with user consent) to improve classification accuracy

---

## 🔍 How It Works

1. **User uploads** or selects a plant image via the app.
2. The image is **sent to a backend server** for analysis.
3. The server responds with:
   - The **detected plant**
   - A **summary of features**
   - Any **warnings or confidence levels**
4. Additional facts about the plant are fetched from **Wikipedia**.



---

## 🔧 Tech Stack

| Technology      | Purpose                      |
|-----------------|------------------------------|
| `Flet`          | UI / frontend                |
| `Firebase`      | Authentication               |
| `Flask`         | Backend API (assumed)        |
| `Wikipedia API` | Factual plant info           |
| `Python`        | Core logic & API integration |


---


## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/DidoeS14/SmartPlantVision.git
cd SmartPlantVision
python -m venv venv
venv\Scripts\activate   # On Windows
source venv/bin/activate  # On Mac/Linux
pip install -r requirements.txt
`````
### 2. Set up the model 
Download the model from [Releases](https://github.com/DidoeS14/SmartPlantVision/releases)
and place it a folder inside the project called models.

### 3. Start the server and the client:
```bash
python src/server.py
python src/main.py
````


### ⚠️ Disclaimer

This application is still work in progress!


---
## 🛠️ Build
To build the client side simply follow the flet build app build tutorial [here](https://flet.dev/docs/publish).
The server doesn't need to be built, it is enough to be ran with python.

## 🔐 Notes on Data Collection
The server collects a subset of user-submitted images inside a folder called "collected_data". This data helps us improve plant classification models and provide better accuracy in future versions.

## 🧪 Example Use Cases
- Gardening apps
- Educational tools
- Agricultural monitoring
- Personal plant diaries

## 🔮 Future Work
Smart PV is a project still in the development stage with several planned improvements to enhance its plant analysis and recommendation capabilities:

- Subtype Classification
  - A dedicated model will be developed to classify specific plant subtypes with greater accuracy.


- Growth Stage Estimation
  - Introduce models to estimate:
  - Time remaining until a plant is fully grown
  - Time passed since it was last considered eatable
  - Current growth stage


- Smart Recommendations
   - Use external sources such as Wikipedia to generate care tips and recommendations for each plant species.


- Expanded Output Fields
  - Future updates may include additional output fields to further enrich plant insights.
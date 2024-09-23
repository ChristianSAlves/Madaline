from django.urls import path
from .views import MadalineTrainView, MadalinePredictView, letras_home

urlpatterns = [
    path('', letras_home, name='letras-home'),
    path('train/', MadalineTrainView.as_view(), name='madaline-train'),
    path('predict/', MadalinePredictView.as_view(), name='madaline-predict'),
]

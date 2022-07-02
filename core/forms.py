from django import forms
from .models import Inputsetone


class InputindForm(forms.ModelForm):
    class Meta:
        model = Inputsetone
        fields = ('tip_doc', 'identificacion', 'pnombre', 'snombre', 'papellido', 'sapellido', 'fecha_nac', 'genero')
        widgets = {
            'tip_doc': forms.Select(attrs={'class': 'form-control'}),
            'identificacion': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'IDENTIFICACIÓN'}),
            'pnombre': forms.TextInput({'class': 'form-control', 'placeholder': 'PRIMER NOMBRE'}),
            'snombre': forms.TextInput({'class': 'form-control', 'placeholder': 'SEGUNDO NOMBRE'}),
            'papellido': forms.TextInput({'class': 'form-control', 'placeholder': 'PRIMER APELLIDO'}),
            'sapellido': forms.TextInput({'class': 'form-control', 'placeholder': 'SEGUNDO APELLIDO'}),
            'fecha_nac': forms.DateInput(format='%Y-%m-%d', attrs={'class': 'form-control', 'type': 'date'}),
            'genero': forms.Select({'class': 'form-control'}),
        }

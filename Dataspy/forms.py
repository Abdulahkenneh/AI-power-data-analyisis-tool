from django import forms
from .models import UserFile

class DataUploadForm(forms.ModelForm):
    uploaded_files = forms.FileField(
        required=False,
        widget=forms.ClearableFileInput(attrs={'multiple': False})  # Allows single file upload
    )
    url = forms.URLField(
        required=False,
        max_length=200,
        widget=forms.TextInput(attrs={'placeholder': 'Enter a URL (optional)'})
    )

    class Meta:
        model = UserFile
        fields = ['uploaded_files']

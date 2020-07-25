from django.shortcuts import render

from .forms import DiagnosisForm

from .models import Diagnosis
# Create your views here.

def diagnosis_create_view(request):
    form = DiagnosisForm(request.POST, request.FILES)
    if form.is_valid():
        form.save()

    context = {'form': form}
    return render(request, "diagnosis.html", context)
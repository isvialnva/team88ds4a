from django.shortcuts import render
from django.views.generic import TemplateView


class HomePageView(TemplateView):

    template_name = "core/core.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        return context


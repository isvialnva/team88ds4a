from django.shortcuts import render
from django.contrib.messages.views import SuccessMessageMixin
from django.views.generic import TemplateView, CreateView, ListView
from .models import Inputsetone
from .forms import InputindForm


class HomePageView(TemplateView):

    template_name = "core/index.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        return context


class AboutView(TemplateView):

    template_name = 'core/about.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        return context


class ProductList(ListView):
    model = Inputsetone
    paginate_by = 10
    context_object_name = 'obj'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        return context


class ProductAdd(SuccessMessageMixin, CreateView):

    template_name = 'core/product.html'
    form_class = InputindForm
    success_message = "Registro guardado correctamente"
    success_url = '/appprestadorgen/asignaprestador/'
    context_object_name = 'obj'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        return context


def productsearch(request):
    doc = request.GET.get('document')
    print(doc)
    return render(request, 'core/searchproduct.html', {'resa': None, 'res': None})


def procesarindividual(request):
    identificacion = request.GET.get('identificacion')
    print(identificacion)
    context = {
        'identificacion': identificacion
    }
    return render(request, "core/procesoind.html", context)

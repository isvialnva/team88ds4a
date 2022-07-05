from django.shortcuts import render, reverse
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
    context_object_name = 'obj'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        return context

    def get_success_url(self):
        return reverse('listproduct')


def productsearch(request):
    doc = str(request.GET.get('document'))
    dat = Inputsetone.objects.filter(identificacion__icontains=doc)
    context = {
        'datos': dat
    }
    return render(request, 'core/searchproduct.html', context)


def procesarindividual(request):
    identificacion = request.GET.get('identificacion')
    print(identificacion)
    context = {
        'identificacion': identificacion
    }
    return render(request, "core/procesoind.html", context)

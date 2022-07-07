from django.urls import path
from .views import HomePageView, AboutView, ProductList, ProductAdd, procesarindividual, productsearch, \
    ProductTwoList, machstring, AddInput, Listcsv, DeleteInput

urlpatterns = [
    path('', HomePageView.as_view(), name='index'),
    path('about', AboutView.as_view(), name='about'),
    path('listproduct', ProductList.as_view(), name='listproduct'),

    # Cargue dos planos csv
    path('listcsv', Listcsv.as_view(), name='listcsv'),
    path('addcsv', AddInput.as_view(), name='addcsv'),
    path('deletecsv/<str:pk>/', DeleteInput.as_view(), name='deletecsv'),


    path('registroadd', ProductAdd.as_view(), name='registroadd'),
    path('search-product', productsearch, name='search-product'),
    path('procesarind', procesarindividual, name='procesarind'),
    path('listproductsinproc', ProductTwoList.as_view(), name='listproductsinproc'),
    path('matchstring', machstring, name='matchstring'),

]

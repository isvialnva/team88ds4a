import pandas as pd
import numpy as np
import recordlinkage
import unicodedata
from django.shortcuts import render, reverse
from django.urls import reverse_lazy
from django.contrib.messages.views import SuccessMessageMixin
from django.views.generic import TemplateView, CreateView, ListView
from django.views.generic.edit import DeleteView
from .models import Inputsetone, Inputsettwo, CargueInput
from .forms import InputindForm, FormCargueInputOne
from proyecto.settings import MEDIA_ROOT

# Renaming Columns
COLUMN_NAMES = {
    'DOC_LOTE': 'id',
    'DOCUMENTO': 'id',
    'NOM1_LOTE': 'name1',
    'NOM1_VAL': 'name1',
    'NOM2_LOTE': 'name2',
    'NOM2_VAL': 'name2',
    'APE1_LOTE': 'lastname1',
    'APE1_VAL': 'lastname1',
    'APE2_LOTE': 'lastname2',
    'APE2_VAL': 'lastname2',
}

# Document corrections
DOCUMENT_REPLACEMENTS = {
    r'[a-zA-Z]': '',  # Removing letters
    r'\b0+': '',  # Removing leading zeros
}

# Words and errors replacements
NAMES_REPLACEMENTS = {
    r'ERR/QUERY': np.nan,  # Remove query error results
    r'&AMP;APOS;': '',  # Remove erroneous value in DB
    r'CORREGIR_': '',  # Remove erroneous value in DB
    r'TRAMITADOR_': '',  # Remove erroneous value in DB
    r'INDETERMINADO': '',  # Remove erroneous value in DB
    r'HIJOS': '',  # Remove Hijos from certain names
    r'VIUDA': '',  # Remove VIUDA from last names
    r'VDA': '',  # Remove VDA (short of VIUDA) from last names
    r'\bDE\b': '',  # Remove DE as part of compound names
    r'\bDEL\b': '',  # Remove DEL as part of compound names
    r'\bLA\b': '',  # Remove LA as part of compound names
    r'\bLAS\b': '',  # Remove LAS as part of compound names
    r'\bLOS\b': '',  # Remove LOS as part of compound names
}

# Character correcions in names and lastnames
CHAR_REPLACEMENTS = {
    r'¥': 'N',
    r'Ç\?': 'N',
    r'Ç ': 'N',
    r'Ç': 'N',
    r'\?': 'N',
    r'Ñ': 'N',
    r'¾': 'N',
    r'V': 'B',
    r'¡': 'J',
    r'Y': 'J',
    r'I': 'J',
    r'LL': 'J',
    r'Z': 'S',
    r'X': 'S',
    r'H': '',
    r'[_\.:,;!&/\|\-\\\+\*\?\'\^\$]': '',
}

# Special characters corrections after removing accents
SPECIAL_CHAR_REPLACEMENTS = {
    r'Y': 'J',  # Remove Gamma
    r'Μ': 'M',  # Remove Miu
}

# Alphabet of final letters used in search
ALPHABET = 'ABCDEFGJKLMNOPQRSTUW'


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
        return reverse('listproductsinproc')


class Listcsv(TemplateView):
    template_name = "core/htmlcsv.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['archivos'] = CargueInput.objects.all().values('id', 'archivo')
        return context


class AddInput(SuccessMessageMixin, CreateView):
    model = CargueInput
    form_class = FormCargueInputOne
    success_message = "Registro guardado correctamente"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        return context

    def get_success_url(self):
        return reverse('listcsv')


class DeleteInput(SuccessMessageMixin, DeleteView):
    model = CargueInput
    success_url = reverse_lazy('listcsv')
    success_message = "Registro eliminado correctamente"


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


class ProductTwoList(ListView):
    model = Inputsettwo
    context_object_name = 'obj'

    def get_queryset(self):
        return Inputsettwo.objects.filter(procesado=False). \
            values('tip_doc', 'identificacion', 'pnombre', 'snombre', 'papellido', 'sapellido', 'fecha_nac', 'genero')


# Algoritmo
def read_data(filename):
    """
    Reads data in csv file and renames columns. CSV file cannot have empty rows at the top.
    """
    df = pd.read_csv(filename, encoding='latin-1')
    df = df.rename(columns=COLUMN_NAMES)
    df['id'] = df['id'].astype(str)
    return df


def strip_accents(word):
    """
    Removes accents from letters.
    """
    return ''.join(l for l in unicodedata.normalize('NFD', word) if unicodedata.category(l) != 'Mn')


def remove_repeated_letters(word):
    """
    Removes repeated consecutive letters.
    """
    output = []
    for l in range(len(word)):
        if l == 0:
            output.append(word[l])
        else:
            if word[l] != word[l - 1]:
                output.append(word[l])
    return ''.join(output)


def clean_data(input_df):
    """
    Cleans input dataframe and returns output dataframe modified.
    """
    output_df = input_df.copy()
    output_df['id'] = output_df['id'].replace(DOCUMENT_REPLACEMENTS, regex=True)  # Document corrections
    output_df = output_df.fillna('')  # Fill nan values as blank

    output_df = output_df.apply(lambda x: x.str.upper())  # Ensures all letters are capitalized.

    output_df = output_df.replace(NAMES_REPLACEMENTS, regex=True)  # Words corrections
    output_df[['name1', 'name2', 'lastname1', 'lastname2']] = output_df[
        ['name1', 'name2', 'lastname1', 'lastname2']].replace('[0-9]', '', regex=True)  # Remove numbers in names
    output_df = output_df.replace(CHAR_REPLACEMENTS, regex=True)  # Character corrections
    output_df[['name1', 'name2', 'lastname1', 'lastname2']] = output_df[
        ['name1', 'name2', 'lastname1', 'lastname2']].applymap(lambda x: strip_accents(str(x)))  # Remove accents
    output_df = output_df.replace(SPECIAL_CHAR_REPLACEMENTS, regex=True)  # Special character corrections
    output_df[['name1', 'name2', 'lastname1', 'lastname2']] = output_df[
        ['name1', 'name2', 'lastname1', 'lastname2']].applymap(
        lambda x: remove_repeated_letters(str(x)))  # Remove consecutive letters

    output_df = output_df.replace(r'[^\w\s]', '', regex=True)  # Remove non alphanumeric symbols
    output_df = output_df.replace(r'\b\w\b', '', regex=True)  # Remove single letter words
    output_df = output_df.replace(r'[\s]', '', regex=True)  # Remove spaces

    output_df = output_df.replace('nan', '')  # Remove null values taken as text

    output_df = output_df.drop_duplicates()  # Remove duplicates

    output_df['id'] = output_df['id'].replace(r'^\s*$', np.nan, regex=True)  # Remove leading spaces in id numbers
    return output_df


def word_to_vec(word, alphabet=ALPHABET):
    """
    Converts a word in a vector with the Alphabet letters according to the number of occurrances of a word.
    """
    alpha_dict = {}
    for l1 in alphabet:
        alpha_dict[l1] = 0
    for l2 in word:
        alpha_dict[l2] = alpha_dict[l2] + 1
    return list(alpha_dict.values())


def preprocess_names(input_df):
    """
    Joins names without spaces, orders letters alphabetically and creates the numeric word vector.
    """
    output_df = input_df.copy()
    output_df['fullname'] = output_df[['name1', 'name2', 'lastname1', 'lastname2']].apply(''.join, axis=1)
    output_df['fullname'] = output_df['fullname'].replace(r'^\s*$', np.nan, regex=True)
    output_df = output_df[output_df.fullname.notna()]
    output_df['fullname_sorted'] = output_df['fullname'].apply(lambda x: ''.join(sorted(x)))
    output_df['fullname_vector'] = output_df['fullname_sorted'].apply(word_to_vec)
    return output_df


def cosine_similarity(list1, list2):
    """
    Calculates the cosine similarity between two lists.
    """
    cos_sim = np.dot(list1, list2) / (np.linalg.norm(list1) * np.linalg.norm(list2))
    return cos_sim


def levenshtein(seq1, seq2):
    """
    Calculates the levenshtein distance between two lists (Tried but not used).
    """
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros((size_x, size_y))
    for x in range(size_x):
        matrix[x, 0] = x
    for y in range(size_y):
        matrix[0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x - 1] == seq2[y - 1]:
                matrix[x, y] = min(
                    matrix[x - 1, y] + 1,
                    matrix[x - 1, y - 1],
                    matrix[x, y - 1] + 1
                )
            else:
                matrix[x, y] = min(
                    matrix[x - 1, y] + 1,
                    matrix[x - 1, y - 1] + 1,
                    matrix[x, y - 1] + 1
                )
    return (matrix[size_x - 1, size_y - 1])


def prepare_df(input_type='multiple', csv_filename=None, doc_id='0', name1='n1', name2='n2', lastname1='n3',
               lastname2='n4'):
    """
    Prepares the dataframe to be used in the search. Returns the original dataframe and the preprocessed one.
    This method can by used for a multiple search by loading a CSV file (same conditions as above) or as a single search by specifying a name to search.
    """
    if input_type == 'single':
        df = pd.DataFrame({
            'ide': doc_id,
            'name1': name1,
            'name2': name2,
            'lastname1': lastname1,
            'lastname2': lastname2,
        })
    elif input_type == 'multiple':
        df = read_data(csv_filename)
    process_df = clean_data(df)
    process_df = preprocess_names(process_df)
    return df, process_df


def match_by_id(val_process_df, ruv_process_df):
    """
    Matches two preprocessed dataframes by the id (document) which is the fist matching criteria.
    As a check, calculates the cosine similarity between the matches.
    Returns a dataframe with the indexes of the matches for each original database and the cosine similarity values.
    Also returns a list of the documents not found in the RUV dataframe.
    """
    cos_sim = []
    indices_val = []
    indices_ruv = []
    ids_found = []
    not_found = []
    for row1 in val_process_df.itertuples():
        for row2 in ruv_process_df.itertuples():
            if row1[1] == row2[1]:
                ids_found.append(row1[1])
                cosine_sim = cosine_similarity(row1[8], row2[8])
                cos_sim.append(cosine_sim)
                indices_val.append(row1[0])
                indices_ruv.append(row2[0])
    not_found = [x for x in val_process_df.id if x not in ids_found]
    results_df = pd.DataFrame(data={'index_val': indices_val, 'index_ruv': indices_ruv, 'cos_sim': cos_sim})
    return results_df, not_found


def match_by_fullname(val_process_df, ruv_process_df, n_results, method):
    """
    Matches two preprocessed dataframes by the fullname using the library recordlinkages.
    Returns a dataframe with the indexes of the matches for each original database.
    For each searched named the output dataframe contains the number of results by name defined as input parameter.
    """
    indexer = recordlinkage.Index()
    indexer.full()
    matching = indexer.index(val_process_df, ruv_process_df)
    print('{} potential matches.'.format(len(matching)))
    ruv_compare = recordlinkage.Compare()
    ruv_compare.string('fullname', 'fullname', label='cos_sim', method=method)
    results_df = ruv_compare.compute(matching, val_process_df, ruv_process_df)

    results_df = results_df.reset_index()
    results_df.columns = ['index_val', 'index_ruv', 'cos_sim']

    filtered_results = pd.DataFrame()
    for group, gr_df in results_df.groupby('index_val'):
        df = gr_df.sort_values(['index_val', 'cos_sim'], ascending=False).head(n_results)
        filtered_results = pd.concat([filtered_results, df])
    return filtered_results


def match_persons(val_process_df, ruv_process_df, n_results=1, method='cosine'):
    """
    Match persons in two preprocessed dataframes. First attempts to search by document, then searches by name.
    The number of results in the fullname search can be defined and the matching method as inputs parameters.
    Defaults are 1 closest match and the cosine similarity method.
    """
    results_id, not_found = match_by_id(val_process_df, ruv_process_df)
    print('{} document coincidences.'.format(int(results_id.shape[0])))
    if len(not_found) > 0:
        print('{} documents not found in RUV. Searching by full name.'.format(int(len(not_found))))
        search_df = val_process_df[val_process_df.id.isin(not_found)]
        results_fullname = match_by_fullname(search_df, ruv_process_df, n_results, method)
        print('Results found. Displaying {} closest results for each name.'.format(n_results))
    return results_id, results_fullname


def retrieve_data_origin(results_df, val_raw_df, ruv_raw_df, threshold):
    """
    Retrieves the result output names from the original dataframes (without preprocessing).
    Returns a dataframe with the names searched and the correspondences by document or closest matches by fullname in the RUV.
    Creates a warning if the cosine similarity value is below the threshold defined as parameter.
    """
    indices_val = results_df['index_val']
    indices_ruv = results_df['index_ruv']
    cos_sim = results_df['cos_sim']

    results_val = val_raw_df.loc[indices_val].reset_index()
    results_val.columns = ['Index_val', 'ID_val', 'Name1_val', 'Name2_val', 'Lastname1_val', 'Lastname2_val']
    results_ruv = ruv_raw_df.loc[indices_ruv].reset_index()
    results_ruv.columns = ['Index_ruv', 'ID_ruv', 'Name1_ruv', 'Name2_ruv', 'Lastname1_ruv', 'Lastname2_ruv']
    results = pd.concat([results_val, results_ruv, cos_sim.reset_index()['cos_sim']], axis=1)
    results['Warning'] = np.where(results.cos_sim < threshold, True, False)
    return results


def find_persons(ruv_raw_df,
                 ruv_process_df,
                 input_type='multiple',
                 csv_filename=None,
                 doc_id='',
                 name1='',
                 name2='',
                 lastname1='',
                 lastname2='',
                 n_results=1,
                 method='cosine',
                 threshold=0.9):
    """
    Main function. Searches the persons in the input dataframe against the RUV dataframe.
    First preprocess the search dataframe, then searches by id and the ids not found are searched by fullname.
    Returns two dataframes, results found by id and results found by fullname.
    """
    val_raw_df, val_process_df = prepare_df(input_type, csv_filename, doc_id, name1, name2, lastname1, lastname2)
    results_id, results_fullname = match_persons(val_process_df, ruv_process_df, n_results, method)

    results_id = retrieve_data_origin(results_id, val_raw_df, ruv_raw_df, threshold)
    results_fullname = retrieve_data_origin(results_fullname, val_raw_df, ruv_raw_df, threshold)
    return results_id, results_fullname


def machstring(request):
    """
    """
    data = CargueInput.objects.all().values('id')
    idar = []
    for dat in data:
        idar.append(dat['id'])
    input1 = CargueInput.objects.get(id=idar[0])
    input2 = CargueInput.objects.get(id=idar[1])

    RUV_db1 = MEDIA_ROOT + '/' + str(input1.archivo)
    inputtwo = MEDIA_ROOT + '/' + str(input2.archivo)

    ruv_raw = read_data(RUV_db1)
    ruv_raw = ruv_raw.fillna('')
    ruv_clean = clean_data(ruv_raw)
    ruv_full = preprocess_names(ruv_clean)
    val_raw, val_full = prepare_df('multiple', inputtwo)
    results_id1, results_fullname1 = find_persons(ruv_raw, ruv_full, 'multiple', inputtwo, n_results=100, threshold=0.9)
    df03rs = results_fullname1.to_html(justify='center',
                                       index=False,
                                       classes='table table-stripped table-sm text-secondary')
    context = {
        'results_id1': df03rs
    }

    return render(request, "core/listmatch.html", context)

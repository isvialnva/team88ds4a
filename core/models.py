from django.core.validators import FileExtensionValidator
from django.core.validators import RegexValidator
import os
from django.db import models


# Create your models here.
class Inputsetone(models.Model):

    TIPDOC = (
        ('CC', 'CEDULA DE CIUDADANIA'),
        ('TI', 'TARJETA DE IDENTIDAD'),
        ('RC', 'REGISTRO CIVIL'),
        ('CE', 'CEDULA DE EXTRAJERIA'),
        ('PA', 'PASAPORTE'),
        ('CP', 'CERTIFICADO DE PERMANENCIA'),
    )

    GEN = (
        ('F', 'FEMENINO'),
        ('M', 'MASCULINO'),
    )

    tip_doc = models.CharField('TIPO DOCUMENTO', max_length=2, choices=TIPDOC, null=True, blank=True)
    identificacion = models.CharField('IDENTIFICACION', max_length=12, db_index=True, null=True, blank=True)
    pnombre = models.CharField('PIMER NOMBRE', max_length=60, null=True, blank=True,
                               validators=[RegexValidator(regex='^[A-Z \xD1]*$', message='Solo letras mayúsculas')])
    snombre = models.CharField('SEGUNDO NOMBRE', max_length=60, null=True, blank=True,
                               validators=[RegexValidator(regex='^[A-Z \xD1]*$', message='Solo letras mayúsculas')])
    papellido = models.CharField('PRIMER APELLIDO', max_length=60, null=True, blank=True,
                                 validators=[RegexValidator(regex='^[A-Z \xD1]*$', message='Solo letras mayúsculas')])
    sapellido = models.CharField('SEGUNDO APELLIDO', max_length=60, null=True, blank=True,
                                 validators=[RegexValidator(regex='^[A-Z \xD1]*$', message='Solo letras mayúsculas')])
    fecha_nac = models.DateField(null=True, blank=True)
    genero = models.CharField('GENERO', max_length=1, choices=GEN, null=True, blank=True)

    objects = models.Manager()

    def __str__(self):
        return self.identificacion

    class Meta:
        ordering = ["identificacion"]


class Inputsettwo(models.Model):

    TIPDOC = (
        ('CC', 'CEDULA DE CIUDADANIA'),
        ('TI', 'TARJETA DE IDENTIDAD'),
        ('RC', 'REGISTRO CIVIL'),
        ('CE', 'CEDULA DE EXTRAJERIA'),
        ('PA', 'PASAPORTE'),
        ('CP', 'CERTIFICADO DE PERMANENCIA'),
    )

    GEN = (
        ('F', 'FEMENINO'),
        ('M', 'MASCULINO'),
    )

    tip_doc = models.CharField('TIPO DOCUMENTO', max_length=2, choices=TIPDOC, null=True, blank=True)
    identificacion = models.CharField('IDENTIFICACION', max_length=12, db_index=True, null=True, blank=True)
    pnombre = models.CharField('PIMER NOMBRE', max_length=60, null=True, blank=True,
                               validators=[RegexValidator(regex='^[A-Z \xD1]*$', message='Solo letras mayúsculas')])
    snombre = models.CharField('SEGUNDO NOMBRE', max_length=60, null=True, blank=True,
                               validators=[RegexValidator(regex='^[A-Z \xD1]*$', message='Solo letras mayúsculas')])
    papellido = models.CharField('PRIMER APELLIDO', max_length=60, null=True, blank=True,
                                 validators=[RegexValidator(regex='^[A-Z \xD1]*$', message='Solo letras mayúsculas')])
    sapellido = models.CharField('SEGUNDO APELLIDO', max_length=60, null=True, blank=True,
                                 validators=[RegexValidator(regex='^[A-Z \xD1]*$', message='Solo letras mayúsculas')])
    fecha_nac = models.DateField(null=True, blank=True)
    genero = models.CharField('GENERO', max_length=1, choices=GEN, null=True, blank=True)
    procesado = models.BooleanField('Procesado', default=False)

    objects = models.Manager()

    def __str__(self):
        return self.identificacion

    class Meta:
        ordering = ["identificacion"]


class CargueInput(models.Model):
    archivo = models.FileField('Archivo de requeridos',
                               upload_to='01_DATA/',
                               validators=[FileExtensionValidator(['csv'])])
    created = models.DateTimeField(auto_now_add=True, verbose_name="Fecha de creación")
    updated = models.DateTimeField(auto_now=True, verbose_name="Fecha de edición")

    objects = models.Manager()

    def delete(self, *args, **kwargs):
        if os.path.isfile(self.archivo.path):
            os.remove(self.archivo.path)
        super(CargueInput, self).delete(*args, **kwargs)

    def __str__(self):
        return self.archivo

    class Meta:
        ordering = ["archivo"]

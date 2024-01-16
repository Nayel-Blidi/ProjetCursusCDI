#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 17:02:21 2023

@author: andro
"""
import os
import cv2
import numpy as np
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from urllib.request import urlretrieve
from selenium import webdriver


#%%

class myExtractor:
    
    def __init__(self, file_name):
        
        self._test_zone = os.path.abspath('') + '/tmp/'
        self._file_path = self._test_zone + file_name
        if not self.checkPath():
            print('Error: Cannot found tmp folder')
        #self._url = url
        self._img_name = 'payload.jpeg'
        self._img_path = ''
        self._pb_type_name = 'bframe.html'
        self._pb_type_path = ''
        self._problem_def = ''
        
    
    
    def checkPath(self):
        return os.path.exists(self._file_path)
    
    
    def foundPathImg(self):
        for root, dirs, files in os.walk(self._test_zone):
            if self._img_name in files:
                return os.path.join(root, self._img_name)
    
    def foundPathPb(self):
        for root, dirs, files in os.walk(self._test_zone):
            if self._pb_type_name in files:
                return os.path.join(root, self._pb_type_name)
    
    
    def foundProblem(self):
        # Importing image and retrieving properties
        image = cv2.imread(self._img_path)
        h, w, d = image.shape
        
        if h == w and w == 450:
            return 'maskImages'
        elif h == w and w == 300:
            return 'multiImages'
        else:
            print('Error: Invalid image format')
            return ''
    
    def foundInstruction(self):
        list_instruction = ['crosswalk',
                            'bicycle',
                            'boats',
                            'bus',
                            'car',
                            'hydrant',
                            'motorcycle',
                            'stairs',
                            'traffic light']
        with open(self._pb_type_path, 'r') as fichier:
            contenu = fichier.read()
            
            occurences = {string: contenu.count(string) for string in list_instruction}
            most_frequent = max(occurences, key=occurences.get)
        
        return most_frequent
    
    
    def extract3x3Image(self):
        # Importing image and retrieving properties
        image = cv2.imread(self._img_path)
        h, w, d = image.shape
        
        # Deducing properties of under-images
        under_h, under_w = h // 3, w // 3
        
        # Retrieving under-images
        list_images = []        #cv2.imshow('result', image)
        for i in range(3):
            for j in range(3):
                under_image = image[i * under_h:(i + 1) * under_h,
                                    j * under_w:(j + 1) * under_w,
                                    :]
                list_images.append(under_image)
        
        return list_images
    
    
    def extract4x4Image(self):
        # Returning in the same format that the 3x3 extractor
        list_images = [cv2.imread(self._img_path)]
        return list_images
    
    
    def extractProblem(self):
        
        # Downloading page
        #self.compilePage()
    
        # Checking image exists
        self._img_path = self.foundPathImg()
        if self._img_path == '':
            print('Error: Image does not exists in the specified folder')
        
        # Checking image exists
        self._pb_type_path = self.foundPathPb()
        if self._pb_type_path == '':
            print('Error: File does not exists in the specified folder')
        
        # Finding problem type
        problem_type = self.foundProblem()
        problem_instruction = self.foundInstruction()
        
        # Retrieving images according to problem type
        list_images = []
        if problem_type == 'maskImages':
            list_images = self.extract4x4Image()
        
        elif problem_type == 'multiImages':
            list_images = self.extract3x3Image()
        
        
        
        return problem_type, list_images, problem_instruction


    def compilePage(self):
        url = self._file_path
        print(url)
        # Créer une instance du navigateur Firefox (le chemin vers GeckoDriver n'est pas nécessaire ici)
        driver = webdriver.Firefox()
        # Accéder à l'URL local
        driver.get(url)
        # Attendre que la page soit complètement chargée (vous pouvez ajuster le délai selon vos besoins)
        driver.implicitly_wait(10)
        # Récupérer le code HTML après le chargement complet
        html_content = driver.page_source
        # Fermer le navigateur
        driver.quit()
        # Enregistrer le code HTML dans un fichier
        with open("page_compiled.html", "w", encoding="utf-8") as html_file:
            html_file.write(html_content)
"""
    def downloadPage(self):
        # Télécharger le contenu de la page
        response = requests.get(self._url)
        html_content = response.text
    
        # Créer un objet BeautifulSoup pour analyser le HTML
        soup = BeautifulSoup(html_content, 'html.parser')
    
        # Télécharger les ressources externes
        for tag in soup.find_all(['img', 'link', 'script']):
            if 'src' in tag.attrs:
                resource_url = tag['src']
            elif 'href' in tag.attrs:
                resource_url = tag['href']
            else:
                continue
    
            absolute_url = urljoin(self._url, resource_url)
            self.downloadResource()
    
        # Enregistrer le HTML
        with open(os.path.join(self._path, 'index.html'), 'w', encoding='utf-8') as html_file:
            html_file.write(html_content)
    
    def downloadResource(self):
        # Télécharger la ressource et l'enregistrer localement
        filename = os.path.basename(urlparse(self._url).path)
        local_path = os.path.join(self._path, filename)
        print(self._path)
        urlretrieve(self._url, self._path)
        print(f"Ressource téléchargée : {self._url}")
    
    


if __name__ == "__main__":
    # Spécifier l'URL de la page à enregistrer
    page_url = "https://www.example.com"

    # Spécifier le dossier de sortie
    output_folder = "output_folder"

    # Créer le dossier de sortie s'il n'existe pas
    os.makedirs(output_folder, exist_ok=True)

    # Télécharger la page web et ses ressources
    download_page(page_url, output_folder)
"""
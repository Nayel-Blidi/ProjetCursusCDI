#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 17:02:21 2023

@author: andro
"""
import os
import cv2


class myExtractor:
    
    def __init__(self, file_name):
        
        self._test_zone = os.path.abspath('') + '/tmp/'
        self._file_path = self._test_zone + file_name
        if not self.checkPath():
            print('Error: Cannot found tmp folder')
        self._img_name = 'payload.jpeg'
        self._img_path = ''
        self._pb_type_name = 'bframe.html'
        self._pb_type_path = ''
        self._problem_def = ''
        
    
    
    def checkPath(self):
        """
        Check if the file path exists. Return a boolean if yes or not it 
        exists.

        Returns
        -------
        TYPE: Boolean
            DESCRIPTION: If the path exists

        """
        return os.path.exists(self._file_path)
    
    
    def foundPathImg(self):
        """
        Found the path to the image(s) of the problem. Return the path to this
        image(s).

        Returns
        -------
        TYPE: String
            DESCRIPTION: Path to the image problem

        """
        # Seaching in the folder tmp
        for root, dirs, files in os.walk(self._test_zone):
            if self._img_name in files:
                return os.path.join(root, self._img_name)
    
    
    def foundPathPb(self):
        """
        Found the path to the html page of the problem. Return the path to this
        html page.

        Returns
        -------
        TYPE: String
            DESCRIPTION: Path to the html containing the problem

        """
        # Seaching in the folder tmp
        for root, dirs, files in os.walk(self._test_zone):
            if self._pb_type_name in files:
                return os.path.join(root, self._pb_type_name)
    
    
    def foundProblem(self):
        """
        Deducing the problem type from image format.

        Returns
        -------
        TYPE: String
            DESCRIPTION: Problem type

        """
        # Importing image and retrieving properties
        image = cv2.imread(self._img_path)
        h, w, d = image.shape
        
        # reCaptcha has two format for the two differents problems
        if h == w and w == 450:
            return 'maskImages'
        elif h == w and w == 300:
            return 'multiImages'
        else:
            print('Error: Invalid image format')
            return ''
    
    
    def foundInstruction(self):
        """
        Finding the problem instruction by counting the key words occurence. It
        return the most likely problem instruction.

        Returns
        -------
        most_frequent : String
            DESCRIPTION: Problem instruction to solve

        """
        # List of instruction in English
        # TODO: Complete this list if you want with other languages
        list_instruction = ['crosswalk',
                            'bicycle',
                            'boats',
                            'bus',
                            'car',
                            'hydrant',
                            'motorcycle',
                            'stairs',
                            'traffic light']
        
        # Counting occurence of each word in the html file
        with open(self._pb_type_path, 'r') as file:
            content = file.read()
            occurences = {string: content.count(string) for string in list_instruction}
            
            # Deducing that the most frequent is the problem instruction
            most_frequent = max(occurences, key=occurences.get)

        return most_frequent
    
    
    def extract3x3Image(self):
        """
        Returning list of images for multiImages problem.

        Returns
        -------
        list_images : list of np.array
            DESCRIPTION: All images of the problem

        """
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
        """
        Returning list of image for maskImages problem.

        Returns
        -------
        list_images : list of np.array
            DESCRIPTION: Image of the problem

        """
        # Returning in the same format that the 3x3 extractor
        list_images = [cv2.imread(self._img_path)]
        return list_images
    
    
    def extractProblem(self):
        """
        Scraping html page and its ressources to deducing problem type,
        instruction and associated images.

        Returns
        -------
        problem_type : String
            DESCRIPTION: maskImages or multiImages
        list_images : List of np.array
            DESCRIPTION: Associated images to the problem
        problem_instruction : String
            DESCRIPTION: Instruction of the problem

        """    
        # Checking image exists
        self._img_path = self.foundPathImg()
        if self._img_path == '':
            print('Error: Image does not exists in the specified folder')
        
        # Checking html page exists
        self._pb_type_path = self.foundPathPb()
        if self._pb_type_path == '':
            print('Error: File does not exists in the specified folder')
        
        # Finding problem type
        problem_type = self.foundProblem()
        
        # Finding problem instruction
        problem_instruction = self.foundInstruction()
        
        # Retrieving images according to problem type
        list_images = []
        if problem_type == 'maskImages':
            list_images = self.extract4x4Image()
        
        elif problem_type == 'multiImages':
            list_images = self.extract3x3Image()
        
        return problem_type, list_images, problem_instruction



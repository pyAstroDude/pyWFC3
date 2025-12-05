#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 10:25:05 2024

@author: sshenoy
"""

import os
import sys
from glob import glob
from datetime import datetime

import pyds9 as ds9
import numpy as np
from astropy.io import fits

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont#, QPixmap
from PyQt5.QtWidgets import (
    QMainWindow,
    QApplication,
    QDesktopWidget,
    QHBoxLayout,
    QVBoxLayout,
    QGridLayout,
    QPushButton,
    QLabel,
    QFrame,
    # QLineEdit,
    QWidget,
    # QDialog,
    # QDialogButtonBox,
    QFileDialog
    )

import pandas as pd


class MainWindow(QMainWindow):

    DESCRIPTION = """
    Class used to view the downloaded *rate.jpg file and select the file to 
    be used as input to generate flats. If there are lot of files to view 
    then the user can view a subset of files and continue with the rest at 
    a latter time. This fclass tags the view status and the selection 
    status. Files once viewed will not be view at a latter time by default 
    but the user can override it.  
    """

    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("Select MIRI Imager Data")
        
        self.set_kwargs()
        
        self.initUI()
    
    def set_kwargs(self):
        if len(sys.argv) == 1:
            self.set_input_directory()
        else:
            self.inpaths = sys.argv[1:]
        
        self.df = self.get_data_frame(self.inpaths)
        
        self.num_files = len(self.df)
        
        self.cur_index = 0
        self.index_str = "File {} of {}"
        
        self.cur_file = self.df['Filename'][self.cur_index].split('/')[-1]
        self.file_str = "Filename: {}"
        print(self.cur_file)
        if 'rate' in self.cur_file:
            self.kind = {'PROD_TYPE': 'rate'}
        elif 'flt' in self.cur_file:
            self.kind = {'PROD_TYPE': 'flt'}
        else:
            self.kind = {'PROD_TYPE': 'NA'}
            
        self.kind['FORMAT'] = self.cur_file.split('.')[-1]
        
        if (self.kind['PROD_TYPE']=='rate') and (self.kind['FORMAT']=='fits'):
            self.fits_head = self.get_fits_headers(self.df['Filename'][self.cur_index])
            
            self.cur_ngrp = self.df['NGroups'][self.cur_index]
            self.cur_xoff = self.df['DithXoff'][self.cur_index]
            self.cur_yoff = self.df['DithYoff'][self.cur_index]
        else:
            self.cur_ngrp = np.nan
            self.cur_xoff = np.nan
            self.cur_yoff = np.nan
       
        self.cur_view = self.df['Viewed'][self.cur_index]
        
        self.cur_stat = self.df['Selected'][self.cur_index]
        self.view_str = "Viewed: {} {} Selected: {}"
        
        
        self.ngrp_str = "NGroups: {} {} DOffset: {:.3f}, {:.3f}"
        
        self.outpath = self.set_output()
        
    
    def initUI(self):
        
        sizeObject = QDesktopWidget().screenGeometry(-1)
        self.setGeometry(int(sizeObject.width())-320, 0, 400, 400)
        
        widget = QWidget()
        self.setCentralWidget(widget)
        
        layout1 = QHBoxLayout()
        layout2 = QVBoxLayout()
        # layout3 = QVBoxLayout()
        
        info_layout = QGridLayout()
        info_layout.setAlignment(Qt.AlignTop)
        info_layout.setContentsMargins(12, 12, 12, 12)
        info_layout.setSpacing(44)
        
        info_label = QLabel("File Information")
        info_label.setAlignment(Qt.AlignCenter)
        info_label.setFont(QFont("Helvetica", 18, QFont.Bold))
        self.info_label = info_label
        
        indx_label = QLabel(self.index_str.format(self.viewed+self.cur_index+1,
                                                  self.num_files))
        
        self.indx_label = indx_label
        
        file_label = QLabel(self.file_str.format(self.cur_file))
        self.file_label = file_label
        
        view_label = QLabel(self.view_str.format(self.cur_view, 
                                                 44 * " ", 
                                                 self.cur_stat))
        self.view_label = view_label
        
        ngrp_label = QLabel(self.ngrp_str.format(self.cur_ngrp,
                                                 44 * " ",
                                                 self.cur_xoff, 
                                                 self.cur_yoff))
        self.ngrp_label = ngrp_label
        
        info_layout.addWidget(self.info_label)
        info_layout.addWidget(self.indx_label)
        info_layout.addWidget(self.file_label)
        info_layout.addWidget(self.view_label)
        info_layout.addWidget(self.ngrp_label)
        
        frame = QFrame()
        frame.setFrameShape(QFrame.StyledPanel | QFrame.Raised)
        
        
        frame.setLayout(info_layout)
        
        layout2.addWidget(frame)
        
        next_button = QPushButton("Next")
        next_button.clicked.connect(self.next_clicked)
        self.next_button = next_button
        
        prev_button = QPushButton("Previous")
        prev_button.clicked.connect(self.prev_clicked)
        self.prev_button = prev_button
        
        save_button = QPushButton("Save Selection")
        save_button.clicked.connect(self.save_clicked)
        
        slct_button = QPushButton("Select")
        slct_button.clicked.connect(self.slct_clicked)
        self.slct_button = slct_button
        
        uslt_button = QPushButton("Unselect")
        uslt_button.clicked.connect(self.uslt_clicked)
        self.uslt_button = uslt_button
        
        maft_button = QPushButton("Save Manifest")
        maft_button.clicked.connect(self.maft_clicked)
        
        quit_button = QPushButton("Quit", self)
        quit_button.clicked.connect(self.close_all)
        
        button_layout = QGridLayout()
        button_layout.setAlignment(Qt.AlignHCenter | Qt.AlignBottom)
        button_layout.addWidget(uslt_button, 0, 1)
        button_layout.addWidget(prev_button, 1, 0)
        button_layout.addWidget(slct_button, 1, 1)
        button_layout.addWidget(next_button, 1, 2)
        button_layout.addWidget(save_button, 2, 0)
        button_layout.addWidget(quit_button, 2, 1)
        button_layout.addWidget(maft_button, 2, 2)
        
        layout2.addLayout(button_layout)
        layout1.addLayout(layout2)
        
        # imag_label = QLabel()
        self.ds9 = ds9.DS9()
        if self.df['Filename'][self.cur_index].split('.')[1] == 'jpg':
            self.ds9.set(f"jpeg {self.df['Filename'][self.cur_index]}")
        elif self.df['Filename'][self.cur_index].split('.')[1] == 'fits':
            self.ds9.set(f"file {self.df['Filename'][self.cur_index]}")
            
        self.ds9.set("zoom to fit")
        
        widget.setLayout(layout1)
        
        self.setCentralWidget(widget)

       
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_S or event.key() == Qt.Key_Down:
            self.slct_clicked()
        elif event.key() == Qt.Key_P or event.key() == Qt.Key_Left:
            self.prev_clicked()
        elif event.key() == Qt.Key_N or event.key() == Qt.Key_Right:
            self.next_clicked()
        elif event.key() == Qt.Key_U or event.key() == Qt.Key_Up:
            self.uslt_clicked()
            
    
    def set_input_directory(self):
        open_dlg = QFileDialog()
        inpath = open_dlg.getExistingDirectory(self, "Load data Directory")
        
        self.inpaths = [inpath]
        
        # return self.inpaths
        
        
    def set_output(self, dirname=None):
        if dirname is None:
            outpath = os.getcwd()
        else:
            outpath = os.path.join(os.getcwd(), dirname)
        
        if not os.path.isdir(outpath):
            os.makedirs(outpath)
        
        return outpath

    
    def get_data_frame(self, inpaths):
        
        filelist = []
        for path in inpaths:

            pth = os.path.abspath(path)
            
            if os.path.isdir(pth):
                flist = glob(os.path.join(pth, '*rate.fits'))
                
                if len(flist) == 0:
                    flist = glob(os.path.join(pth, '*rate.jpg'))
                    if len(flist) == 0:
                        print("Input directory does not contain")
                        print("fits or jpg rate files.")
                        flist = glob(os.path.join(pth, '*flt.fits'))
                        
                        if len(flist) == 0:
                            print("Input directory does not contain")
                            print("fits FLT files. Existing.....")
                            sys.exit()
                        kind = {'FORMAT': 'fits', 'PROD_TYPE': 'flt'}
                    kind = {'FORMAT': 'jpg', 'PROD_TYPE': 'rate'}
                
                    
                kind = {'FORMAT': 'fits', 'PROD_TYPE': 'rate'}
                
                flist.sort()
                filelist.extend(flist)
                
                if (kind['FORMAT']=='fits') & (kind['PROD_TYPE']=='rate'):
                    in_dict = {"Filename": filelist,
                               "Viewed": [False for fl in filelist],
                               "Selected": [False for fl in filelist],
                               "NGroups": ["" for fl in filelist],
                               "ExpNums": ["" for fl in filelist],
                               "DithPatt": ["" for fl in filelist],
                               "DithTotl": ["" for fl in filelist],
                               "DithNum": ["" for fl in filelist],
                               "DithXoff": ["" for fl in filelist],
                               "DithYoff": ["" for fl in filelist],
                               }
                else:
                    in_dict = {"Filename": filelist,
                               "Viewed": [False for fl in filelist],
                               "Selected": [False for fl in filelist],
                               }
                
                df = pd.DataFrame(in_dict)
                self.viewed = 0
            elif os.path.isfile(pth):
                df_unsorted = pd.read_csv(pth)
                if (not 'Filename' in df_unsorted.columns):
                    df_unsorted = pd.read_csv(pth, header=None)
                    if len(df_unsorted.columns) == 1:
                        df_unsorted['Filename'] = df_unsorted[0].apply(os.path.abspath)
                        df_unsorted.drop(0, axis=1, inplace=True)
                    else:
                        print("More than one column in the input file.")
                        print("Please update csv file header and re-try the command.")
                        sys.exit()
                
                req_cols = ['Viewed', 'Selected']
                for col_name in req_cols:
                    if not col_name in df_unsorted.columns:
                        df_unsorted[col_name] = False
                
                df = df_unsorted.sort_values(by=["Viewed"]).reset_index(drop=True).copy()
                self.viewed = len(df["Viewed"][df["Viewed"]==True]) - 1
            else:
                print("\n Input directory or file not found.")
                print(" Existing ......")
                sys.exit()
        
        return df
    
    
    def get_fits_headers(self, filename):
        
        fits_hdr = {'Filepath': [],
                    'Filename': [],
                    'NGroups': [],
                    'ExpNums': [],
                    'DithPatt': [],
                    'DithTotl': [],
                    'DithNum': [],
                    'DithXoff': [],
                    'DithYoff': [],
                    }
        
        fullfile = self.df['Filename'][self.cur_index]
        
        if os.path.isfile(fullfile):
            fname = fullfile
        else:
            fname = filename
        
        fits_hdr['Filepath'].append(fname)
        
        
        if fname.split('.')[-1] != 'fits':
            hdr_ngrp = "Non-FITS File"
            hdr_dpth = "Non-FITS File"
        else:
            try:
                hdr = fits.getheader(fname)
            except FileNotFoundError:
                hdr_ngrp = "FITS FnF"
                hdr_dpth = "FITS FnF"
            
            hdr_ngrp = hdr['NGROUPS']
            hdr_nexp = hdr['EXPOSURE']
            try:
                hdr_dpth = hdr['PATTTYPE']
            except KeyError:
                hdr_dpth = 'KW Missing'
            hdr_dtot = hdr['NUMDTHPT']
            hdr_dnum = hdr['PATT_NUM']
            hdr_xoff = hdr['XOFFSET']
            hdr_yoff = hdr['YOFFSET']
            
        if fname.split('/')[-1] == self.cur_file:
            fits_hdr['Filename'] = self.cur_file
            fits_hdr['NGroups'].append(hdr_ngrp)
            fits_hdr['ExpNums'].append(hdr_nexp)
            fits_hdr['DithPatt'].append(hdr_ngrp)
            fits_hdr['DithTotl'].append(hdr_dtot)
            fits_hdr['DithNum'].append(hdr_dnum)
            fits_hdr['DithXoff'].append(hdr_xoff)
            fits_hdr['DithYoff'].append(hdr_yoff)
        
        self.df.loc[self.cur_index, 'NGroups'] = hdr_ngrp
        self.df.loc[self.cur_index, 'ExpNums'] = hdr_ngrp
        self.df.loc[self.cur_index, 'DithPatt'] = hdr_dpth
        self.df.loc[self.cur_index, 'DithTotl'] = hdr_dtot
        self.df.loc[self.cur_index, 'DithNum'] = hdr_dnum
        self.df.loc[self.cur_index, 'DithXoff'] = hdr_xoff
        self.df.loc[self.cur_index, 'DithYoff'] = hdr_yoff
    
    
    def update_image(self):
        self.cur_istr = self.index_str.format(str(self.viewed+self.cur_index+1),
                                              self.num_files)
        self.cur_file = self.df['Filename'][self.cur_index].split('/')[-1]
        self.cur_fstr = self.file_str.format(self.cur_file)
        self.cur_vstr = self.view_str.format(self.df['Viewed'][self.cur_index],
                                             44 * " ",
                                             self.df['Selected'][self.cur_index])
        
        if (self.kind['PROD_TYPE']=='rate') and (self.kind['FORMAT']=='fits'):
            self.fits_head = self.get_fits_headers(self.df['Filename'][self.cur_index])
            
            self.cur_nstr = self.ngrp_str.format(self.df['NGroups'][self.cur_index],
                                                 44 * " ",
                                                 self.df['DithXoff'][self.cur_index],
                                                 self.df['DithYoff'][self.cur_index])
        else:
            self.cur_nstr = self.ngrp_str.format(np.nan,
                                                 44 * " ",
                                                 np.nan,
                                                 np.nan)
        
        self.indx_label.setText(self.cur_istr)
        self.file_label.setText(self.cur_fstr)
        self.view_label.setText(self.cur_vstr)
        self.ngrp_label.setText(self.cur_nstr)
        
        if self.df['Filename'][self.cur_index].split('.')[1] == 'jpg':
            self.ds9.set(f"jpeg {self.df['Filename'][self.cur_index]}")
        elif self.df['Filename'][self.cur_index].split('.')[1] == 'fits':
            self.ds9.set(f"file {self.df['Filename'][self.cur_index]}")
        
        self.ds9.set("zoom to fit")
        
    
    def on_text_changed(self):
        new_inx_str = self.indx_label.text()
        new_idx = int(new_inx_str.split(' ')[1]) - 1
        if new_idx >= 0 and new_idx <= self.num_files - 1:
            self.cur_index = new_idx
            self.update_image()
        else:
            msg = "\n Invaid index: {}".format(int(new_inx_str.split(' ')[1]))
            print(msg)
            print(" Index should be between {} and {}".format(1,self.num_files))
        

    def next_clicked(self):
        self.df.loc[self.cur_index, 'Viewed'] = True
        if self.cur_index < self.num_files - 1:
            self.cur_index = self.cur_index + 1
            self.update_image()
        else:
            print("\n This is the last file. Please use the previous ")
            print(" button (or Key p/left-arrow) to view files")
        
        
    def prev_clicked(self):
        self.df.loc[self.cur_index, 'Viewed'] = True
        if self.cur_index > 0:
            self.cur_index = self.cur_index - 1
            self.update_image()
        else:
            print("\n You are already at the first file. Please use")
            print(" the next button (or Key n/right-arrow) to view files.")


    def slct_clicked(self):
        self.df.loc[self.cur_index, 'Selected'] = True
        self.next_clicked()
        
        
    def uslt_clicked(self):
        self.df.loc[self.cur_index, 'Selected'] = False
        self.next_clicked()
        
        
    def save_clicked(self, usr_input=True):
        time_str = datetime.now().strftime("%Y-%m-%dT%H%M%S")
        
        tmp_fname = os.path.join(self.outpath, 
                                 "selected_input_"+time_str+".csv")
        
        print("\n Saving selected data to csv file...")
        
        if usr_input is False:
            self.df.to_csv(tmp_fname, index=False)
        else:
            fl_dlg = QFileDialog()
            out_fname, _ = fl_dlg.getSaveFileName(self, "Save Select File",
                                                      tmp_fname)
            if out_fname == "":
                print("\n\tFile Not Saved. Exiting.....")
            else:
                self.df.to_csv(out_fname, index=False)
            
        
    def maft_clicked(self):
        mani_df = self.df.copy()
        
        if self.kind['FORMAT'] == 'jpg':
            fname_col = mani_df['Filename'].str.replace('jpg', 'fits')#.str.replace('rate', 'uncal')
        if self.kind['PROD_TYPE'] == 'rate':
            fname_col = mani_df['Filename'].str.replace('rate', 'uncal')
        if self.kind['PROD_TYPE'] == 'flt':
            fname_col = mani_df['Filename'].str.replace('flt', 'raw')
        
        mani_lst =list(fname_col[mani_df['Selected']==True])
        
        fname = os.path.join(self.outpath, 'manifest.lst')
        
        fl_dlg = QFileDialog()
        outfile, _ = fl_dlg.getSaveFileName(self, "Save Manifest", fname)
        
        if outfile == "":
            print("\n Manifest Not Saved.")
        else:
            with open(outfile, 'w+') as f:
                for line in mani_lst:
                    f.write(line+'\n')
    
    
    def close_all(self):
        try:
            self.ds9.set("exit")
        except ValueError:
            print("DS9 is not running. Exiting....")
        self.save_clicked()
        QApplication.quit()
        
        
    def closeEvent(self, event):
        try:
            self.ds9.set("exit")
        except ValueError:
            print("DS9 is not running. Exiting....")
        self.save_clicked()
        event.accept()  
        


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

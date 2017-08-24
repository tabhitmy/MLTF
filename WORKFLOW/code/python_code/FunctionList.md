NFDALauncher.py

* [PATH]
* controlPanel()
    * processCodeEncoder()
        * iterCodePool()
* mainProcesser(process\_code)
    * processCodeDecoder()
    * labelReading()
    * featurePreparation()
    * dataConstrution()
    * datasetSeparation() 
    * #######{
        * caseLOO()
        * caseLPO()
        * caseKFold()
        * caseStratifidKFold()
        * caseRawsplit()
            * train\_test\_constructor()
     ######}
    * labelProcessor()
    * featurePlotting()
    * feaSelection()
    * ####### {
        * tSNE()
        * normalPCA()
    	######}
    
    * dataBalance()
    * ###### {
        * downSamplingNega()
    	######}


* ####{
    * sklearnTrainer()
    * dataSetPreparation()
    * ######{
        * svmLinear()
        * svmKernal()
        * lda()
        * qda()
        * naivebayes()
        * adaboost()
        * sgdClassifier()
        * logiRegression()
        * decisionTree()
        * randomForest()
            * dataRegulationSKL()
                * ko1processor()
            * sci-kit Learn Default Functions...
            * processLearning()
                * calculateFRAP()
                    * fScore()
    	######}

            
    * kerasTrainer()        
    * 'Not in detail at this moment Aug 24th,2017 '
 #####}

    * resultPlotting

* 'result summary screening in command line format. Only numerical results. '

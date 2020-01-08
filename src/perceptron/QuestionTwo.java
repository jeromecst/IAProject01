package perceptron;

import java.util.Random;
import mnisttools.MnistReader;
import java.util.Arrays;
import java.io.FileWriter;
import java.io.IOException;



public class QuestionTwo {
    /* Les donnees */
    public static String path=perceptronMulti.path;
    public static String labelDB=perceptronMulti.labelDB;
    public static String imageDB=perceptronMulti.imageDB;

    /* Parametres */
    // Nv exemples pour l'ensemble d'évaluation
    public static final int Nv = 1000;
    public static final int Na = 1000;
    public static final int Nt = 1000;
    // Nombre d'epoque max
    public final static int EPOCHMAX=40;
    // Classe positive (le reste sera considere comme des ex. negatifs):
    public static int  classe = 12 ;
    public static float eta = (float) 0.003; // valeur initiale
    // Générateur de nombres aléatoires
    public static int seed = 1234;
    public static Random GenRdm = new Random();

    public static MnistReader db = new MnistReader(labelDB, imageDB);
    /* Tableau où stocker les données */
    public static int longueur=db.getImage(1)[1].length;
    public static int largeur=db.getImage(1).length;

    /* Donnée d'apprentissage */


    public static void main(String[] arg)  throws IOException {
        float[][] trainData= new float[Na][largeur*longueur+1];
        int[] label = new int[Na];

        float[][] validData= new float[Nv][largeur*longueur+1];
        int[] labelVal = new int[Nv];

        float[][] testData= new float[Nt][largeur*longueur+1];
        int[] labelTest = new int[Nt];

        int labelInt, j, o, epoc, minErrVal, minErrValTot=1000;
        float minEta = (float) 1;
        // on rempli les tableaux de l'ensemble de validation à 1000
        o = 0;
        j = 1;
        while(j < Nv){
            o += 1;
            labelInt = db.getLabel(o);
            if (labelInt >= 10 && labelInt <= 21){
                labelVal[j]=labelInt;
                validData[j]= perceptronMulti.ConvertImage(perceptronMulti.BinariserImage(db.getImage(o),(int)255/2));
                j += 1;
            }
        }


        // on rempli les tableaux de l'ensemble d'apprentissage à 1000
        j = 1;
        while( j < Na + 1){
            o += 1;
            labelInt = db.getLabel(o);
            if (labelInt >= 10 && labelInt <= 21){
                label[j-1] = labelInt;
                trainData[j-1] = perceptronMulti.ConvertImage(perceptronMulti.BinariserImage(db.getImage(o),255/2));
                j += 1;
            }
        }

        // on rempli les tableaux de l'ensemble de test à 1000
        j = 1;
        while( j < Nt + 1){
            o += 1;
            labelInt = db.getLabel(o);
            if (labelInt >= 10 && labelInt <= 21){
                labelTest[j-1] = labelInt;
                testData[j-1] = perceptronMulti.ConvertImage(perceptronMulti.BinariserImage(db.getImage(o),255/2));
                j += 1;
            }
        }

        float [][] poids = perceptronMulti.InitialiseW(largeur*longueur+1); //poids[class][donnée xi]
        float[][] result = new float[Na][classe]; //result[Na][class] récupere les probalités que Na est dans la classe
        float[][] resultVal = new float[Nt][classe];
        j = 1;
        o = 0;
        int err, errVal;
        int maxInd, maxIndVal;

        FileWriter fw = new FileWriter("questionTwo");
        while(eta < 0.1){
            poids = perceptronMulti.InitialiseW(largeur*longueur+1);

            epoc = 0;
            minErrVal=1000;
            eta += (float) 0.001;
            do {
                err = 0;
                errVal = 0;
                for (int i = 0; i < Na; i++) { //i<Na
                    result[i] = perceptronMulti.InfPerceptron(trainData[i], poids); 
                    perceptronMulti.Majpoids(trainData[i], poids, eta, result[i], label[i]); // Mise à jour des poids sur l'ensemble d'apprentissage
                    maxInd = perceptronMulti.MaxIndiceTab(result[i]); // On récupère l'indice max, c'est à dire le label obtenu pour la donnée

                    if(i < Nv){
                        resultVal[i] = perceptronMulti.InfPerceptron(validData[i], poids);
                        maxIndVal = perceptronMulti.MaxIndiceTab(resultVal[i]); // Mise à jour des poids sur l'ensemble d'apprentissage
                        // On cherche le nombre d'erreurs sur l'ensemble de validation
                        if (labelVal[i] - 10 != maxIndVal ) {
                            errVal += 1;
                        }
                    }

                    // On cherche le nombre d'erreurs sur l'ensemble d'apprentissage
                    if (label[i] - 10 != maxInd) {
                        err += 1;
                    }

                }
                System.out.println(epoc + " (" + err + "~" + errVal + ") " + eta + " >= " + minEta); // debug
                epoc += 1;

                if(errVal < minErrVal){ // on récupère le minimum sur l'erreurs de validation global
                    minErrVal = errVal;
                }

            } while (epoc < EPOCHMAX && err > (int)(.005*Na));
            // à moins de 5% d'erreurs, on stoppe l'aglorithme

            if(minErrVal < minErrValTot){
                minErrValTot = minErrVal;
                minEta = eta;
            }

            fw.write(eta+" "+minErrVal);
            fw.write("\n");

        }
        fw.close();



        FileWriter ff = new FileWriter("questionTwo2");
        System.out.println("Ensemble de teste avec minEta = " + minEta);
        epoc=0;
        poids = perceptronMulti.InitialiseW(largeur*longueur+1); //re-initialisation des poids

        // perceptron sur l'ensemble de test
        do {
            err = 0;
            errVal = 0;
            for (int i = 0; i < Nt; i++) { //i<Na
                result[i] = perceptronMulti.InfPerceptron(testData[i], poids); //Initialise les results
                perceptronMulti.Majpoids(testData[i], poids, minEta, result[i], labelTest[i]);  // on oublie pas d'utiliser minEta

                maxInd = perceptronMulti.MaxIndiceTab(result[i]);
                if (labelTest[i] - 10 != maxInd) {
                    err += 1;
                }

                resultVal[i] = perceptronMulti.InfPerceptron(validData[i], poids); //Initialise les results
                maxIndVal = perceptronMulti.MaxIndiceTab(resultVal[i]);
                if (labelVal[i] - 10 != maxIndVal ) {
                    errVal += 1;
                }
            }

            System.out.println(epoc + " (" + err + "~" + errVal + ") ");
            ff.write(epoc+" "+err+" "+errVal);
            ff.write("\n");
            epoc+=1;


        } while (err > 0);
        // on arrête lorsqu'il n'y a plus d'erreurs

        ff.close();
    }
}

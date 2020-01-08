package perceptron;

import java.awt.Color;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Random;

import javax.imageio.ImageIO;

import mnisttools.MnistReader;

public class QuestionSix {
    /* Les donnees */
    public static String path=perceptronMulti.path;
    public static String labelDB=perceptronMulti.labelDB;
    public static String imageDB=perceptronMulti.imageDB;

    /* Parametres */
    // Na exemples pour l'ensemble d'apprentissage
    public static final int Na = 5000;
    // Nv exemples pour l'ensemble d'évaluation
    public static final int Nv = 1000;
    // Nombre d'epoque max
    public final static int EPOCHMAX=50;
    // Classe positive (le reste sera considere comme des ex. negatifs):
    public static int  classe = 12 ;
    public static float eta = (float) 0.05;
    // Générateur de nombres aléatoires
    public static int seed = 1234;
    public static Random GenRdm = new Random();

    public static MnistReader db = new MnistReader(labelDB, imageDB);
    /* Tableau où stocker les données */
    public static int longueur=db.getImage(1)[1].length;
    public static int largeur=db.getImage(1).length;

    /* Donnée d'apprentissage */
    public static float[][] trainData= new float[Na][largeur*longueur+1];
    public static float[][] validData= new float[Nv][largeur*longueur+1];
    public static int[] label = new int[Na];
    public static int[] labelVal = new int[Nv];
    
    
    public static int[][][] Tabimage = new int[Nv][largeur][longueur]; 
    public static void main(String[] arg)  throws IOException {
       
        int labelInt, j, o, epoc = 0;
        o = 0;
        j = 0;
        
        while( j < Na ){
            o += 1;
            labelInt = db.getLabel(o);
            if (labelInt >= 10 && labelInt <= 21){
                label[j] = labelInt;
                trainData[j] = perceptronMulti.ConvertImage(perceptronMulti.BinariserImage(db.getImage(o),(int)255/2));
                j += 1;
            }
        }

        o = 697932;
        j = 0;
        // on rempli l'ensemble de validation en partant de la fin de la base de donnée
        while(j < Nv){
            o -= 1;
            labelInt = db.getLabel(o);
            if (labelInt >= 10 && labelInt <= 21){
                labelVal[j]=labelInt;
                validData[j]= perceptronMulti.ConvertImage(perceptronMulti.BinariserImage(db.getImage(o),(int)255/2));
                Tabimage[j] = db.getImage(o);

                j += 1;
            }
        }

        float[][] poids = perceptronMulti.InitialiseW(largeur*longueur+1); //poids[class][donnée xi]
        float[][] result = new float[Na][classe]; //result[Na][class] récupere les probalités que Na est dans la classe

        float[][] resultVal = new float[Nv][classe];

        float[] MalClasse = new float[Nv];
        int maxindval;
        int minindBC;
        do{
            for(int i=0; i<Na;i++) { //i<Na
                result[i]=perceptronMulti.InfPerceptron(trainData[i], poids); //Initialise les results
                perceptronMulti.Majpoids(trainData[i], poids, eta, result[i], label[i]);
            }
            epoc+=1;
           System.out.println(epoc);

        } while(epoc<EPOCHMAX);
        
        for(int i=0; i<MalClasse.length;i++) {
        	MalClasse[i]=1; // On cherchera les images avec le moins bon score d'inf�rence
        }
        
        for(int i=0; i<Nv; i++) {
            resultVal[i]=perceptronMulti.InfPerceptron(validData[i], poids); //Initialise les results
            maxindval = perceptronMulti.MaxIndiceTab(resultVal[i]);
            if(maxindval != labelVal[i]-10) {
            	MalClasse[i] = resultVal[i][maxindval];
            }
        }
        
        epoc=0;
        int labelMC;
        char[] Caractere = new char[classe]; 
        String A = "ABCDEFGHIJKL";
        Caractere=A.toCharArray();
        
        while(epoc<5){
        	minindBC = perceptronMulti.MinIndiceTab(MalClasse);
        	 // On la sauvegarde
        	labelMC=labelVal[minindBC]-10;
        	int[][] image = Tabimage[minindBC];
            int numberOfColumns = 28;//image.length;
            int numberOfRows = 28; //image[0].length;
            BufferedImage bimage = new BufferedImage(numberOfColumns, numberOfRows, BufferedImage.TYPE_BYTE_GRAY);

            for(int i=0; i<28; i++) {
                for(j=0; j<28; j++) {	
                     int c = image[i][j]; // ici 0 pour noir, 255 pour blanc
                     int rgb = new Color(c,c,c).getRGB();
                     bimage.setRGB(j,i,rgb);
                }
            }
            
            // enregistrement
            if(labelMC == 1 || labelMC ==2 || labelMC ==4 || labelMC == 5 || labelMC ==8) {//On cherche les classes C,B,E,F et I
	            File outputfile = new File(path+"QuestionSixImages"+Caractere[labelMC]+".png");
	            if(! outputfile.exists()) { //On ne veut pas deux images de la meme classe
		            ImageIO.write(bimage, "png", outputfile);
		            epoc+=1;
	            }
            }
            MalClasse[minindBC] = 1;//On cherche le min

        }
       
        System.out.print(0);
       
    }

}
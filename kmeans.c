/* IMPLEMENATION OF K-MEANS FOR PURE DATA: UNSUPERVISIONED MACHINE LEARNING ALGORITHM
 * FOR DATA CLUSTERIZATION.
 *  
 *  by Adriano Monteiro, december, 27, 2012. contact: monteiro.adc@gmaill.com
 *  
 */

#include "m_pd.h"
#include "stdlib.h"
#include "math.h"


static t_class *kmeans_class;

typedef struct _kmeans {
        t_object x_obj;
        t_int dataSize;
        t_int maxData;
        t_int nData;
        t_atom *list;
        
        t_int clusters_number; 
        t_atom *means;
        t_int gate; 
        
        t_atom *class;
        t_atom *cost;
        t_float costMean;
        t_float precision;
        t_int iterations;
        
        t_outlet *finalCost, *extern_class;
        
}t_kmeans;

/*********************** INTERNAL FUNCTIONS  *******************************/

//it inserts in data buffer elements received from left inlet. 
void kmeans_list(t_kmeans *x, t_symbol *s, int argc, t_atom *argv) {
     

		float aux;
		t_atom *adress;
		int i = 0;
		
		if (x->dataSize != argc) {
             error("Input list must have the same length than previous ones."); 
        }else if (x->nData >= x->maxData){
              post("Maximum of data was achieved.");
         } else {
         	adress = x->list; 
            adress = adress + (x->nData * x->dataSize);
             while(i < argc){
                  aux = atom_getfloat(argv);
                  SETFLOAT(adress, aux); 
                  argv++; adress++; i++;  
              }  
           x->nData++; 
                                         
          }
}

// random generator without repetition
int unique(int max, int *buf,int pos){
	
	int try, chek, k;
	
	
	if(pos == 0){
		try = (int)(rand()%max);
		buf[pos]= try;
		return try;
		} else{
			try= (int)(rand()%max);
			chek =0;
			for(k =0; k < pos; k++){
				chek = chek + (int)(try==buf[k]);
				}
			if(chek == 0){
				buf[pos]=try; 
				return try;
				}else{ 
					return unique(max, buf, pos);  
					}
		 }
}

// it initializes clusters' means by choosing them from data set. 
void random_ini(t_kmeans *x){
	
	int i = 0, j=0, k=0, pos= 0;
	
	int *buf = (int *)getbytes(sizeof(int )* x->clusters_number);
	for(i=0; i < x->clusters_number; i++) buf[i]=0;
	
	for (i=0; x->clusters_number > i; i++){
		pos = unique(x->nData, buf,i) * x->dataSize;
		//post("position = %d", pos);   AQUI!!!!
		for(j=0; j < x->dataSize; j++){
			x->means[k] = x->list[pos+j];
			k++;
		}
	}
	freebytes(buf, sizeof(int)*x->clusters_number);
}

// it classifies each data according their distances from clusters' mean.
void classify(t_kmeans *x){
	
	int i, j, k, pos_d, pos_m, inc_m, inc_d ;
	float acum = 0, minor, index;
	x->costMean =0;
	
	
	for(i = 0; x->nData > i; i++){	
		pos_d	= i * x->dataSize;
	 for(k =0; x->clusters_number > k; k++){
		pos_m	= k * x->dataSize; 
		for(j = 0; x->dataSize > j; j++){
			inc_m = pos_m + j;
			inc_d = pos_d + j;
			acum = pow(atom_getfloat(x->means+inc_m)- atom_getfloat(x->list+inc_d), 2.0)+ acum;	
			}
		acum = sqrt(acum);
		if(k == 0 || acum<minor){minor=acum;index = k;}
		acum =0;
	 	}
	 SETFLOAT(x->class+i, index);
	 SETFLOAT(x->cost+i, minor);
	}
	
	// it calculates the mean cost.
	for(i = 0; x->nData > i; i++) x->costMean = x->costMean + atom_getfloat(x->cost+i);
	x->costMean = x->costMean / x->nData;
	
}

// it calculates new clusters'means.
void new_means(t_kmeans *x){
	
	int i, j, pos_m, pos_l, cluster;
	float  value;
	t_float *div;
	
	// create a buffer for counting clusters
	div = (t_float *)getbytes(sizeof(t_float )* x->clusters_number);
	for(i=0; x->clusters_number > i; i++){ div[i] = 0;}
	
  	// it sums the values of every element indexed to each cluster 	
	for(i=0; x->nData > i; i++){
		cluster = (int)atom_getfloat(x->class+i);
		div[cluster] = div[cluster]+1; // and count the number of elements in each class
		for(j=0; x->dataSize > j; j++){
			pos_m = x->dataSize*cluster+j;
			pos_l = x->dataSize*i+j;
			if(div[cluster]==1){
				SETFLOAT(x->means+pos_m, atom_getfloat(x->list+pos_l));
			}else {
				value = atom_getfloat(x->means+pos_m)+ atom_getfloat(x->list+pos_l);
		    	SETFLOAT(x->means+pos_m, value);
				}
			}
		}
	
	// finally, it divides the sum of each value by their counts to get the means
	for (i=0; x->clusters_number > i; i++){
		for(j=0; x->dataSize > j; j++){
			pos_m = x->dataSize*i+j;
			value = atom_getfloat(x->means+pos_m);
			if(value != 0 && div[i] != 0){
			value = atom_getfloat(x->means+pos_m) / div[i];
			SETFLOAT(x->means+pos_m, value);
			}
		}
	}
	
	// just freeing the counting buffer
	freebytes(div, sizeof(t_float)*x->clusters_number);
	
}


/********************** INTERFACE FUNCTIONS *************************************/

// normalize data in buffer.
void kmeans_normalize(t_kmeans *x){
	
	int i, total = x->maxData * x->dataSize;
	float max=0, value;		
	
	for(i=0; i<total; i++){
		value = atom_getfloat(x->list+i);
		if(i==0 || max < value)max = value;
	}
	for(i=0; i<total; i++){
		value = atom_getfloat(x->list+i)/max;
		SETFLOAT(x->list+i, value);
	}
}
// return all element from data buffer. 
void kmeans_dump(t_kmeans *x){
	int i, pos;
	t_atom *dump;
	
	for(i=0; x->nData > i;i++ ){
	pos = i*x->dataSize;	
	dump = x->list + pos;
	outlet_list(x->x_obj.ob_outlet, gensym("list"), x->dataSize, dump);	
	}
}

// get one especific element from data buffer.
void kmeans_get(t_kmeans *x, t_floatarg element){
	if(element<x->nData){
	t_atom *saida = x->list+ (int)(x->dataSize * element);
	outlet_list(x->x_obj.ob_outlet, gensym("list"), x->dataSize, saida);
	} else error("Index %d out of range.", (int)element);
}

// pseudo-clear process - disables acess to buffers.
void kmeans_clear(t_kmeans *x){
	
	x->nData = 0;
	x->gate = 0;
}

// chance configurations for 'element size' and 'maximum number of elements' inside data buffer.
void kmeans_reset(t_kmeans *x, t_floatarg size, t_floatarg maximum){
	
	kmeans_free(x);
	x->dataSize = (int)size;
	x->maxData= (int)maximum;
	int total = (int)size * (int)maximum, means = size * x->clusters_number;
	x->list = (t_atom *)getbytes(sizeof(t_atom )* total);
	x->class = (t_atom *)getbytes(sizeof(t_atom )* maximum);
	x->cost = (t_atom *)getbytes(sizeof(t_atom )* maximum);
	x->means = (t_atom *)getbytes(sizeof(t_atom )* means);
	
	x->nData = 0;
	x->gate = 0;
	
}
// change the 'maximum number of elements' inside data buffer. 
void kmeans_max(t_kmeans *x, t_floatarg maximum){
	
	int i , minor, max = x->dataSize * maximum;
	
	t_atom *newList = (t_atom *)getbytes(sizeof(t_atom )* max);
	t_atom *newClass = (t_atom *)getbytes(sizeof(t_atom )* maximum);
	t_atom *newCost = (t_atom *)getbytes(sizeof(t_atom )* maximum);
	
	if(maximum <= x->nData){
		minor=maximum; 
		x->nData = maximum;
	} else minor = x->nData;
	for(i=0; minor*x->dataSize > i; i++)newList[i] = x->list[i];
	for(i=0; minor > i; i++){newClass[i] = x->class[i]; newCost[i] = x->cost[i];}

	freebytes(x->list, (int)x->dataSize * (int)x->maxData);
	freebytes(x->cost, (int)x->maxData);
	freebytes(x->class, (int)x->maxData);
	x->list = newList;
	x->class = newClass;
	x->cost = newCost;
	x->maxData = (int)maximum;
}
// change the number of clusters to find. 
void kmeans_clusters(t_kmeans *x,t_floatarg newSize){
	
	freebytes(x->means, (x->dataSize*x->clusters_number) * sizeof(t_atom));
	x->clusters_number = newSize;
	x->means = (t_atom *)getbytes((x->dataSize*newSize) * sizeof(t_atom ));
	x->gate=0;

}
// return the means of each cluster. 
void kmeans_get_clusters_means(t_kmeans *x){
	
	int i, pos;
	t_atom *dump;
	if(x->gate==1){
		for(i=0; x->clusters_number > i ;i++ ){
		pos = i*x->dataSize;	
		dump = x->means + pos;
		outlet_list(x->x_obj.ob_outlet, gensym("list"), x->dataSize, dump);	
		}
	}
}

// change precision of difference bettween consecutive elements in cost funtion
void kmeans_precision(t_kmeans *x, t_floatarg precision){
	x->precision = precision;	
}

// change the number of iterations for optimization. 
void kmeans_iterations(t_kmeans *x, t_floatarg iterations){
	x->iterations = iterations;	
}

/******************* BANG MESSAGE - CLUSTERIZATION ***************************************/

void kmeans_bang(t_kmeans *x) {
	
	if(x->nData == 0){post("Empty");}
	else if(x->nData < x->clusters_number){error("Number of input data must be equal or greater than number of clusters.");}
	else{
		t_float antCost, cost;
		int i, j, meansSize = x->clusters_number*x->dataSize;
		t_atom *saida = (t_atom *)getbytes(sizeof(t_atom )* x->nData);
		t_atom *selMeans = (t_atom *)getbytes(sizeof(t_atom )* meansSize);
		
			// optimize cost in function of number of iterations.
			for(i= 0; x->iterations >= i; i++) {
				random_ini(x);	
				classify(x);	
				do{ antCost = x->costMean;
					new_means(x);
					classify(x);
					}while(antCost - x->costMean > x->precision);

				if(i == 0 || x->costMean < cost){
					cost = x->costMean;
					for(j = 0; x->nData > j; j++)saida[j] = x->class[j];
					for(j = 0; meansSize > j; j++)selMeans[j] = x->means[j];
					} 
				}
		freebytes(x->means, meansSize * sizeof(t_atom));
		x->means = selMeans;
		outlet_list(x->x_obj.ob_outlet, gensym("list"), x->nData, saida);
		outlet_float(x->finalCost, cost);
		freebytes(saida, x->nData * sizeof(t_atom));
		x->gate=1;
		}
}

/****************************** RIGHT INLET FUCTION **********************************/

// classify righ inlet input according clusters' means found in the last clusterization of data buffer. 
void kmeans_extern(t_kmeans *x, t_symbol *s, int argc, t_atom *argv){
	
	int k, j, pos_m, inc_m, index;
	float acum, minor;
	
	if(argc != x->dataSize) error("Input list must have the same length as trainig samples in memory.");
	else {
	
		for(k =0; x->clusters_number > k; k++){
				pos_m	= k * x->dataSize; 
				for(j = 0; x->dataSize > j; j++){
					inc_m = pos_m + j;
					acum = pow(atom_getfloat(x->means+inc_m)- atom_getfloat(argv+j), 2.0)+ acum;	
					}
				acum = sqrt(acum);
				if(k == 0 || acum<minor){minor=acum;index = k;}
				acum =0;
				}
		
		outlet_float(x->extern_class, index);
		outlet_float(x->finalCost, minor);
	}
}


/************************** OBJECT CREATION AND SETUP *************************************/

void *kmeans_new(t_symbol *s, int argc, t_atom *argv){
     t_kmeans *x = (t_kmeans *)pd_new(kmeans_class);
     
	switch(argc){
	case 4:
		x->dataSize = atom_getfloat(argv);
		x->maxData = atom_getfloat(argv+1);
		x->clusters_number = atom_getfloat(argv+2);
		x->iterations = atom_getfloat(argv+3);
		break;	
	case 3:
		x->dataSize = atom_getfloat(argv);
		x->maxData = atom_getfloat(argv+1);
		x->clusters_number = atom_getfloat(argv+2);
		x->iterations = 0;
		break;
	case 2:
		x->dataSize = atom_getfloat(argv);
		x->maxData = atom_getfloat(argv+1);
		x->clusters_number = 2;
		x->iterations = 0;
		break;
	case 1:
		x->dataSize = atom_getfloat(argv);
		x->maxData = 10;
		x->clusters_number = 2;
		x->iterations = 0;
		break;
	case 0:
		x->dataSize = 1;
		x->maxData = 10;
		x->clusters_number = 2;
		x->iterations = 0;;
	}
	
	 x->gate = 0;
     x->nData = 0;
	 x->precision = 0.01;
	 x->means = (t_atom *)getbytes(sizeof(t_atom )* (int)(x->clusters_number * x->dataSize));
     x->list = (t_atom *)getbytes(sizeof(t_atom )* (int)(x->maxData * x->dataSize));
     x->class = (t_atom *)getbytes(sizeof(t_atom )* (int) x->maxData);
     x->cost = (t_atom *)getbytes(sizeof(t_atom )* (int) x->maxData);
     
     inlet_new(&x->x_obj, &x->x_obj.ob_pd, gensym("list"), gensym(""));
     
     outlet_new(&x->x_obj, 0);
     x->finalCost = outlet_new(&x->x_obj, gensym("float"));
     x->extern_class = outlet_new(&x->x_obj, gensym("float"));
     
     return (void *) x;
}

void kmeans_free(t_kmeans *x){
	
	int total = x->maxData * x->dataSize, means = x->dataSize * x->clusters_number;
	freebytes(x->list, sizeof(t_atom)* total);
	freebytes(x->means, sizeof(t_atom)* means);
	freebytes(x->class, sizeof(t_atom)* x->maxData);
	freebytes(x->cost, sizeof(t_atom)* x->maxData);
}
    
void kmeans_setup(void) {
     kmeans_class = class_new(gensym("kmeans"), (t_newmethod)kmeans_new, 
     (t_method)kmeans_free, sizeof(t_kmeans), CLASS_DEFAULT, A_GIMME, 0);
      
     class_addlist(kmeans_class, kmeans_list);         
     class_addbang(kmeans_class, kmeans_bang);
     class_addmethod(kmeans_class, (t_method)kmeans_clear, gensym("clear"),0);
     class_addmethod(kmeans_class, (t_method)kmeans_reset, gensym("reset"), A_DEFFLOAT, A_DEFFLOAT, 0);
     class_addmethod(kmeans_class, (t_method)kmeans_max, gensym("max"), A_DEFFLOAT, 0);
     class_addmethod(kmeans_class, (t_method)kmeans_get, gensym("get"), A_DEFFLOAT, 0);
     class_addmethod(kmeans_class, (t_method)kmeans_clusters, gensym("clusters"), A_DEFFLOAT, 0);
     class_addmethod(kmeans_class, (t_method)kmeans_dump, gensym("dump"), 0);
     class_addmethod(kmeans_class, (t_method)kmeans_get_clusters_means, gensym("get_clusters_means"), 0);
     class_addmethod(kmeans_class, (t_method)kmeans_precision, gensym("precision"), A_DEFFLOAT, 0);
     class_addmethod(kmeans_class, (t_method)kmeans_iterations, gensym("iterations"), A_DEFFLOAT, 0);
     class_addmethod(kmeans_class, (t_method)kmeans_normalize, gensym("normalize"), 0);
     class_addmethod(kmeans_class, (t_method)kmeans_extern, gensym(""), A_GIMME, 0);
}


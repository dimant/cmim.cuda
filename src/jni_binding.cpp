#include "weka_attributeSelection_CMIMAttributeEvaluator.h"

#define BFSCUDA_EXPORTS
#include "bfscuda.h"


class JNIRowIterator : public RowIterator {
private:
	JNIEnv* env;
	jobject object;
	jobject instances;

	jclass jEnumeration;
	jmethodID jMID_Enumeration_hasMoreElements;
	jmethodID jMID_Enumeration_nextElement;

	jclass jWekaInstance;
	jmethodID jMID_WekaInstance_value;
	jmethodID jMID_WekaInstance_numAttributes;

public:
	JNIRowIterator(JNIEnv *_env, jobject _object, jobject _instances) : 
	  env(_env), 
	  object(_object),
	  instances(_instances) {
		jEnumeration = env->FindClass("java/util/Enumeration");
		jMID_Enumeration_hasMoreElements = env->GetMethodID(jEnumeration, "hasMoreElements", "()Z");
		jMID_Enumeration_nextElement  = env->GetMethodID(jEnumeration, "nextElement", "()Ljava/lang/Object;");

		jWekaInstance = env->FindClass("weka/core/Instance");
		jMID_WekaInstance_value = env->GetMethodID(jWekaInstance, "value", "(I)D");
		jMID_WekaInstance_numAttributes = env->GetMethodID(jWekaInstance, "numAttributes", "()I");
	  }

	void get_row(std::vector<std::string>& row) {
		jint i;
	
		row.clear();
		char str_value[32];

		jboolean flag = env->CallBooleanMethod(instances, jMID_Enumeration_hasMoreElements);

		if( flag ) {
			jobject instance = env->CallObjectMethod(instances, jMID_Enumeration_nextElement);
			jint d = env->CallIntMethod(instance, jMID_WekaInstance_numAttributes);

			jdouble value;
			jboolean iscopy;

			for(i = 0; i < d; i++) {
				value = env->CallDoubleMethod(instance, jMID_WekaInstance_value, i);

				sprintf(str_value, "%g", (double) value);
				row.push_back(std::string(str_value));

				// row push_back classIndex
			}
		}

		return;
	}
};

JNIEXPORT jfloatArray JNICALL Java_weka_attributeSelection_CMIMAttributeEvaluator__1process(
		JNIEnv *env, 
		jobject object, 
		jobject instances) 
{

	int i, n;

	std::vector<struct result_record*> result_vector;
	Data data;

	JNIRowIterator jniRowIterator(env, object, instances);

	std::cout << "Loading Data... " << std::endl;
	loadData(data, jniRowIterator);

	std::cout << "Calculating CMIM... " << std::endl;
	process(data, 2, result_vector);

	std::cout << "Returning Results..." << std::endl;
	n = result_vector.size();

	jfloatArray jresult_array = env->NewFloatArray(n);
	jfloat* jresult_buffer = (jfloat*) malloc(sizeof(jfloat) * n);

	for(i = 0; i < n; i++) {
		jresult_buffer[i] = result_vector[i]->i_u_yx;
	}
	
	env->SetFloatArrayRegion(jresult_array, 0, n, jresult_buffer);

	free(jresult_buffer);

	std::cout << "Done." << std::endl;
	return jresult_array;
}



/* ADD THIS TO THE HTML FILE:
<script src='https://cdnjs.cloudflare.com/ajax/libs/mathjs/2.7.0/math.min.js
'></script>
*/

// Input values as a javascript array with 1 as the first value

var predvalue = function(input){

	if (typeof input[8] == 'undefined' && typeof(input[9]) == 'undefined') {
		
		input.pop();
		input.pop();

		return predbasemodel(input);

	} else if (typeof input[8] != 'undefined' && typeof(input[9]) == 'undefined') {

		input.pop()
		return predwcholesterol(input);

	} else if (typeof input[8] == 'undefined' && typeof(input[9]) != 'undefined') {

		input.splice(8,1)
		return predwbp(input)

	} else {

		return predwbpch(input)

	}

}


var predbasemodel = function(input){
	// Input format: [1,age,sex,thalach,exang,years(smoker at least 2 years),famhist,thalrest]
	var coeff = [ 0.10500203,0.03503578, 1.44883298, -0.0301142, 1.37564255, 0.1144102, 0.58716087, 0.00487661];
	var xt = math.multiply(coeff,input)
	return 1/(1+Math.pow(Math.E,-xt))
}

//console.log(predbasemodel([1,45,1,185,0,1,1,66]))
// Should output .1950

var predwcholesterol = function(input){
	// Input format: [1,age,sex,thalach,exang,years(smoker at least 2 years),famhist,thalrest,cholesterol]
	var coeff = [-0.1193843, 0.02250782, 1.61838502, -0.03522242, 1.27050383, 0.12327811, 0.52015734, 0.00207758  0.00736791]
	var xt = math.multiply(coeff,input)
	return 1/(1+Math.pow(Math.E,-xt))
}

var predwbp = function(input){
	// Input format: [1,age,sex,thalach,exang,years(smoker at least 2 years),famhist,thalrest,trestbp]
	var coeff = [-0.07929409, 0.02949325,1.41989162,-0.03242003,1.33069392,0.10921246,0.55968885, 0.00338425, 0.01180823]
	var xt = math.multiply(coeff,input)
	return 1/(1+Math.pow(Math.E,-xt))
}

var predwbpch = function(input){
	// Input format: [1,age,sex,thalach,exang,years(smoker at least 2 years),famhist,thalrest,cholesterol,trestbp]s
	var coeff = [-0.1193843, .01929929, 1.59688396, -.03653256, 1.25157194, .115829804, .507826350, .0011524556, .007183951, .00736351577];
	var xt = math.multiply(coeff,input)
	return 1/(1+Math.pow(Math.E,-xt))
}

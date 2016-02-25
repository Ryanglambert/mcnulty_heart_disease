var increment = function() {
    var i = 0;
    return function() { return i += 1; };
};

var ob = increment();

document.addEventListener('DOMContentLoaded', function () {
    document.getElementById('bt2').addEventListener('click', myFunction);
function myFunction() {

    var vall = "";
    var x = '';



    var ele = document.getElementsByName('age_check');


    for (i = 0; i < ele.length; i++) {
        if (ele[i].checked) {
            vall = ele[i].value;

        }
    }
    if (vall == "F" && document.getElementById("mail").value != "") {
        x = document.getElementById("mail").value;
        //return x;
    }
    else if (vall == "T" && document.getElementById("mail").value != "") {
        x = document.getElementById("mail").value;
        // return x;
    }
    else if (vall == "" || document.getElementById("mail").value == "") {
        alert("Please Select All Options");
    }

    //alert('easy');
    //alert(vall);

    if (vall === "T")
    {

        chrome.runtime.sendMessage({ from: 'true', message: 'Information from popup.' });


    }
    else if (vall == "F")
    {

        alert("You are eligbile for the content");
    }
    }
 
});


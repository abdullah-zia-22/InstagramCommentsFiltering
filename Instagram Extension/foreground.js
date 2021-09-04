var checkInterval = 10000;

console.log("Script Running");
alert("Script Running. It can take 5 to 10 seconds");


setInterval(async () => {

	if(checkCommentsDivLength() > 10) {
		runAlgo();
	}

}, checkInterval)

async function runAlgo() {
	var dataToSend = [];
	var comments = getAllNewCommentsInDiv();
	for(var i = 0; i < comments.length; i++) {
		var uuid = createUUID();
		comments[i].setAttribute("data-uuid", uuid)
		dataToSend.push({
			id: uuid,
			text: comments[i].innerText,
			prediction: null
		});
	}
	var predictedComments = await sendRequest(dataToSend);
	dataToSend = [];

	markNewCommentsAsChecked(comments);
	hideOffensiveComments(comments, predictedComments);
}

function checkCommentsDivLength() {
	var commentsWrapper = document.querySelectorAll('.eo2As .EtaWk .XQXOT .Mr508 .C4VMK');
    var allComments = commentsWrapper;
	var comments = []
	for(var i = 0; i < allComments.length; i++) {
		if(!allComments[i].getAttribute("data-uuid") && !allComments[i].getAttribute("data-ischecked")) {
			comments.push(allComments[i]);
		}
	}
	return comments.length;
}

function getAllNewCommentsInDiv() {
    var commentsWrapper = document.querySelectorAll('.eo2As .EtaWk .XQXOT .Mr508 .C4VMK');
    console.log(commentsWrapper);
    var allComments = commentsWrapper;
	var comments = []
	for(var i = 0; i < allComments.length; i++) {
		if(!allComments[i].getAttribute("data-uuid") && !allComments[i].getAttribute("data-ischecked")) {
			comments.push(allComments[i]);
		}
	}
	return comments;
}

function markNewCommentsAsChecked(comments) {
	for(var i = 0; i < comments.length; i++) {
		comments[i].setAttribute("data-ischecked", "true");
	}
}

function hideOffensiveComments(comments, predictedComments) {
	for(var i = 0; i < comments.length; i++) {
		if(comments[i].getAttribute("data-uuid") === predictedComments[i].id &&
			predictedComments[i].prediction === "Offensive"
			) {
				// comments[i].style.display = "none";
			comments[i].style.backgroundColor = "#a9ccbd";
			comments[i].style.color = "#a9ccbd";
			comments[i].style.border = "thin dotted red"
		}
	}
}

async function sendRequest(dataToSend) {
	return new Promise(async (resolve, reject) => {
		try {
			const rawResponse = await fetch('https://fyp-nlp.herokuapp.com/comments_prediction', {
				method: 'POST',
				headers: {
				'Accept': 'application/json',
				'Content-Type': 'application/json'
				},
				body: JSON.stringify(dataToSend)
			});
			var predictedComments = await rawResponse.json();
			resolve(predictedComments)
		}
		catch(error) {
			console.log(err)
			reject(err);
		}
	})
}

function createUUID(){
    var dt = new Date().getTime();
    var uuid = 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
        var r = (dt + Math.random()*16)%16 | 0;
        dt = Math.floor(dt/16);
        return (c=='x' ? r :(r&0x3|0x8)).toString(16);
    });
    return uuid;
}

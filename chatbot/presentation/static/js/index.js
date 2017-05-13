var $messages = $('.messages-content'),
    d, h, m,
    i = 0;

$(window).load(function() {
  $messages.mCustomScrollbar();
  $('.message-input').val(null);
  $('.message-output').val(null);

  setTimeout(function() {
    //fakeMessage();
  }, 100);
});

function updateScrollbar() {
  $messages.mCustomScrollbar("update").mCustomScrollbar('scrollTo', 'bottom', {
    scrollInertia: 10,
    timeout: 0
  });
}



function insertMessage() {
  msg = $('.message-input').val();
  if ($.trim(msg) == '') {
    return false;
  }
  $('<div class="message message-personal">' + msg + '</div>').appendTo($('.mCSB_container')).addClass('new')
  //$('.message-input').val(null);
  updateScrollbar();
	interact(msg);
  setTimeout(function() {
    //fakeMessage();
  }, 1000 + (Math.random() * 20) * 100);
}

$('.message-submit').click(function() {
  insertMessage();
});
$('.message-reset').click(function() {
  $('.message-input').val(null);
  $('.message-output').val(null);
});

$(window).on('keydown', function(e) {
  if (e.which == 13) {
    insertMessage();
    return false;
  }
})


function interact(message){
	// loading message
  //$('<div class="message loading new"><figure class="avatar"><img src="/static/res/botim.png" /></figure><span></span></div>').appendTo($('.mCSB_container'));
	// make a POST request [ajax call]
	$.post('/message', {
		msg: message,
	}).done(function(reply) {
		// Message Received
		// 	remove loading meassage
    $('.message.loading').remove();
		// Add message to chatbox
    //$('<div class="message new">' + reply['text'] + '</div>').appendTo($('.mCSB_container')).addClass('new');
    $('.message-output').val(reply['text']);
    updateScrollbar();

		}).fail(function() {
				alert('error calling function');
				});
}

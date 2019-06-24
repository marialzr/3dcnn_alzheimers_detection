function prezice() {
    document.querySelector('.bg-modal').style.display = 'none';
    document.querySelector('.modal-opac-signup').style.display = 'none';
    document.querySelector('.bg-modal').style.display = 'flex';
    document.querySelector('.modal-opac').style.display = 'flex';
    document.querySelector('.hidediv').style.display = 'none';
}

function train() {
    document.querySelector('.bg-modal').style.display = 'none';
    document.querySelector('.modal-opac').style.display = 'none';
    document.querySelector('.bg-modal').style.display = 'flex';
    document.querySelector('.modal-opac-signup').style.display = 'flex';
}

function show() {
    document.querySelector('.imagini').style.display = 'inline-block';
    document.querySelector('.hidediv').style.display = 'inline-block';

}

function hide() {
    document.querySelector('.imagini').style.display = 'none';
    document.querySelector('.hidediv').style.display = 'none';
}

function custompredict() {
    document.querySelector('.custompredict').style.display = 'inline-block';
}

function modal() {
    document.querySelector('.bg-modal').style.display = 'none';
    document.querySelector('.modal-opac').style.display = 'none';
    document.querySelector('.bg-modal').style.display = 'flex';
    document.querySelector('.modal-opac-signup').style.display = 'none';
    document.querySelector('.modal-opac-signupp').style.display = 'flex';
    document.querySelector('.custompredict').style.display = 'none';
}


document.querySelector('.close').addEventListener('click',
    function () {
        document.querySelector('.bg-modal').style.display = 'none';
        document.querySelector('.modal-opac').style.display = 'none';
        document.querySelector('.imagini').style.display = 'none';
    }
);

document.querySelector('.closesign').addEventListener('click',
    function () {
        document.querySelector('.bg-modal').style.display = 'none';
        document.querySelector('.modal-opac-signup').style.display = 'none';
    }
);

document.querySelector('.closepredict').addEventListener('click',
    function () {
        document.querySelector('.bg-modal').style.display = 'none';
        document.querySelector('.modal-opac-signupp').style.display = 'none';
    }
);

$('#file-upload').change(function () {
    var i = $(this).prev('label').clone();
    var file = $('#file-upload')[0].files[0].name;
    $(this).prev('label').text(file);
});

$('#another-file-upload').change(function () {
    var i = $(this).prev('label').clone();
    var file = $('#another-file-upload')[0].files[0].name;
    $(this).prev('label').text(file);
});
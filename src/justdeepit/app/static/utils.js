var moduleRunningStatus = {'mode': null, 'status': null, 'timestamp': null};
var focusedInputField = null;
var msgSelectFile = 'Please select a FILE or add a file name after the selected folder.';
var msgSelectFolder = 'Please select a FOLDER';
var msgLeftBank = 'Left bank if no ____ dataset';
var module = location.pathname.split('/')[1] === 'module' ?  location.pathname.split('/')[2] : null;


function node2path(node) {
    let dir_path = '';
    if (node.parent !== null) {
        dir_path = dir_path + node2path(node.parent) + '/';
    }
    dir_path = dir_path + node.name;
    return dir_path;
}


function node2filetype(node) {
    let fileType = null;
    if ((node.children.length === 0) && (!node.isEmptyFolder)) {
        fileType = 'file';
    } else {
        fileType = 'folder';
    }
    return fileType;
}


function refreshParams() {
    let deferred = new $.Deferred;
    $.ajax({
        type: 'GET',
        url: '/api/config',
        data: {module: module},
        dataType: 'json',
        cache: false,
    }).done(function(formFields, textStatus, jqXHR) {
        for (let formId in formFields) {
            for (let fieldId in formFields[formId]) {
                let fieldId_ = '#module--' + formId + '--' + fieldId;
                if ($(fieldId_)) {
                    if ($(fieldId_).is(':checkbox')) {
                        if (formFields[formId][fieldId]) {
                            $(fieldId_).attr('checked', true).prop('checked', true).change();
                        } else {
                            $(fieldId_).attr('checked', false).prop('checked', false).change();
                        }
                    } else {
                        $(fieldId_).val(formFields[formId][fieldId]);
                    }
                }
            }
        }
        deferred.resolve();
    }).fail(function(XMLHttpRequest, textStatus, errorThrown) {
        deferred.reject();
        console.log("XMLHttpRequest : " + XMLHttpRequest.status);
        console.log("textStatus     : " + textStatus);
        console.log("errorThrown    : " + errorThrown.message);
    });

    return deferred.promise();
}


function refreshModuleTabs() {
    let tabClasses = [null, null, null];
    if (moduleRunningStatus.status === null) {
        if ($('#module--BASE--status').val() !== 'COMPLETED') {
            tabClasses = ['module-tab-active', 'module-tab-disabled', 'module-tab-disabled'];
        } else {
            tabClasses = ['module-tab-active', null, null];
        }
    } else {
        if (moduleRunningStatus.status === 'RUNNING') {
            if (moduleRunningStatus.mode === 'BASE') {
                tabClasses = ['module-tab-active-disabled', 'module-tab-disabled', 'module-tab-disabled'];
            }
            else if (moduleRunningStatus.mode === 'TRAIN') {
                tabClasses = ['module-tab-disabled', 'module-tab-active-disabled', 'module-tab-disabled'];
            }
            else if (moduleRunningStatus.mode === 'INFERENCE') {
                tabClasses = ['module-tab-disabled', 'module-tab-disabled', 'module-tab-active-disabled'];
            }
        } else {
            if (moduleRunningStatus.mode === 'BASE') {
                tabClasses = ['module-tab-active', null, null];
            }
            else if (moduleRunningStatus.mode === 'TRAIN') {
                tabClasses = [null, 'module-tab-active', null];
            }
            else if (moduleRunningStatus.mode === 'INFERENCE') {
                tabClasses = [null, null, 'module-tab-active'];
            }
        }
    }
    tabClasses.forEach(function(tabClass, i) {
        $('.module-tab').eq(i).removeClass('module-tab-disabled module-tab-active-disabled');
        if (tabClass !== null) {
            $('.module-tab').eq(i).addClass(tabClass);
            if (tabClass === 'module-tab-active' || tabClass === 'module-tab-active-disabled') {
                $('.module-tab-content').removeClass('module-tab-content-active').eq(i).addClass('module-tab-content-active');
            }
        }
    });
}


function refreshShutdownButton() {
    if (moduleRunningStatus.status === 'RUNNING') {
        $('#shutdown-panel').hide();
        $('#shutdown').addClass('button-disable');
        $('#shutdown').attr('disabled', true);
    } else {
        $('#shutdown-panel').show();
        $('#shutdown').removeClass('button-disable');
        $('#shutdown').attr('disabled', false);
    }
}


function setSelectedFile(selectFieldId = 'filetree-selected') {
    selectedFilePath = $('#' + selectFieldId).val();
    if (focusedInputField == 'module--TRAIN--model_weight') {
        if (selectedFilePath.slice((selectedFilePath.lastIndexOf(".") - 1 >>> 0) + 2) !== 'pth') {
            selectedFilePath = selectedFilePath + '.pth';
        }
    }
    $('#' + focusedInputField).val(selectedFilePath);
    focusedInputField = null;
    refreshExecuteButtons();
    $('#select-file-popup').fadeOut();
}


function refreshFileSelectButtons() {
    if (moduleRunningStatus.status === 'RUNNING') {
        $('button.select-file').each(function(i) {
            $(this).addClass('button-disable');
            $(this).attr('disabled', true);
        });
        //$('button#config-edition-open').addClass('button-disable');
        //$('button#config-edition-open').attr('disabled', true);
    } else {
        $('button.select-file').each(function(i) {
            $(this).removeClass('button-disable');
            $(this).attr('disabled', false);
        });
        //$('button#config-edition-open').removeClass('button-disable');
        //$('button#config-edition-open').attr('disabled', false);
    }
}



function refreshRunningStatus() {
    let deferred = new $.Deferred;
    $.ajax({
        type: 'GET',
        url: '/api/status',
        dataType: 'json',
        cache: false,
    }).done(function(runningstatus, textStatus, jqXHR) {
        moduleRunningStatus = runningstatus;
        if (moduleRunningStatus.status === 'RUNNING') {
            $('#app-status-control').fadeIn(3000);
        } else {
            $('#app-status-control').fadeOut(1000);
            $('#force-stop').attr('disabled', false).text('FORCE STOP');
        }
        deferred.resolve();
    }).fail(function(XMLHttpRequest, textStatus, errorThrown) {
        deferred.reject();
        console.log("XMLHttpRequest : " + XMLHttpRequest.status);
        console.log("textStatus     : " + textStatus);
        console.log("errorThrown    : " + errorThrown.message);
    });
     return deferred.promise();
}


function refreshRunningLog() {
    $.ajax({
        type: 'GET',
        url: '/api/log',
        dataType: 'text',
        cache: false,
    }).done(function(logData, textStatus, jqXHR) {
        $('#app-log-msg').html(logData.slice(1, -1).replaceAll('\\"', '"'));
    }).fail(function(XMLHttpRequest, textStatus, errorThrown) {
        console.log("XMLHttpRequest : " + XMLHttpRequest.status);
        console.log("textStatus     : " + textStatus);
        console.log("errorThrown    : " + errorThrown.message);
    }).always(function() {
        $('#app-log-msg').animate({
            scrollTop: $('#app-log-msg')[0].scrollHeight - $('#app-log-msg')[0].clientHeight
        }, 1500);
    });
}
 

function refreshExecuteButtons() {
    let forms = ['#module--BASE', '#module--TRAIN', '#module--INFERENCE'];

    for (let formId of forms) {
        let buttonId = formId + '--run';
        let isValid = true;
        $(formId + ' .file-path').each(function() {
            required_field = true;
            if ($(this).attr('id') === 'module--BASE--config') required_field = false;

            if (required_field) { 
                if ($(this).val() === '') {
                    isValid = false;
                    $(this).addClass('highlight-empty-input');
                } else {
                    $(this).removeClass('highlight-empty-input');
                }
            }
        });
        $(formId + ' .digits').each(function() {
            if ($(this).val() === '') {
                isValid = false;
                $(this).addClass('highlight-empty-input');
            } else {
                $(this).removeClass('highlight-empty-input');
            }
        });
        $(formId + ' .text').each(function() {
            if ($(this).val() === '') {
                isValid = false;
                $(this).addClass('highlight-empty-input');
            } else {
                $(this).removeClass('highlight-empty-input');
            }
        });

        if ((isValid) && ((moduleRunningStatus.status === null) || (moduleRunningStatus.status !== 'RUNNING'))) {
            $(buttonId).removeClass('button-disable');
            $(buttonId).attr('disabled', false);
        } else {
            $(buttonId).addClass('button-disable');
            $(buttonId).attr('disabled', true);
        }
        if (formId == '#module--BASE') {
            refreshConfigField();
        }
    }
}


function __refreshModuleDelay() {
    refreshRunningLog();
    refreshModuleTabs();
    refreshFileSelectButtons();
    refreshExecuteButtons();
    refreshShutdownButton();
}


function _refreshModuleDelay() {
    $.when(refreshRunningStatus()).done(__refreshModuleDelay);
}


function refreshModule(force_run=false) {
    if (moduleRunningStatus.status === 'STARTED' || moduleRunningStatus.status === 'RUNNING') {
        force_run = true;
    }
    if (force_run) {
        $.when(refreshParams()).done(_refreshModuleDelay);
    }
}


function refreshConfigField() {
    if ($('#module--BASE--architecture').val() != 'custom') {
        $('#module--BASE--config').val(null);
        $('#module--BASE--config').attr('readonly', true);
        $('#config-select-open').attr('disabled', true);
        $('#config-select-open').addClass('button-disable');
    } else {
        $('#module--BASE--config').attr('readonly', false);
        $('#config-select-open').attr('disabled', false);
        $('#config-select-open').removeClass('button-disable');
    }
}
 

function refreshTrainOptimizer() {
    default_optmizer = '';
    default_scheduler = '';
    if ($('#module').val() === 'SOD') {
        default_optmizer = 'Adam(params, lr=0.001, betas=(0.9, 0.999)';
        default_scheduler = 'ExponentialLR(optimizer, gamma=0.9)';
    } else {
        default_optmizer = 'dict(type="Adam", lr=0.001)';
        default_scheduler = 'dict(policy="CosineAnnealing", warmup="linear", warmup_iters=1000, warmup_ratio=0.1)';
    }
    $('#module--TRAIN--optimizer').attr('placeholder', default_optmizer);
    $('#module--TRAIN--scheduler').attr('placeholder', default_scheduler);
    if ($('#module').val() !== 'SOD') {
        $('#module--TRAIN--optimizer').attr('disabled', false);
        $('#module--TRAIN--scheduler').attr('disabled', false);
    }
}





$(function(){
    // module action tabs
    $('.module-tab').on('click', function() {
        if (!$(this).hasClass('module-tab-disabled')) {
            $('.module-tab-active').removeClass('module-tab-active');
            $(this).addClass('module-tab-active');
            const i = $('.module-tab').index(this);
            if (i === 1) refreshTrainOptimizer(); // set optimizer and scheduler placeholders
            $('.module-tab-content').removeClass('module-tab-content-active')
                                 .eq($('.module-tab').index(this))
                                 .addClass('module-tab-content-active');
        }
    });


    // refresh button status
    $('form').on('change keyup keypress blur mousemove', function() {
        refreshExecuteButtons();
    });


    // load workspace and enable module action tabs
    $('#module--BASE--run').click(function() {
        $.ajax({
            type: 'POST',
            url: '/module/' + module + '/BASE',
            data: $('#module--BASE').serialize(),
            dataType: 'json',
            cache: false,
        }).done(function(x, textStatus, jqXHR) {
            refreshModule(true);
        }).fail(function(XMLHttpRequest, textStatus, errorThrown) {
            console.log("XMLHttpRequest : " + XMLHttpRequest.status);
            console.log("textStatus     : " + textStatus);
            console.log("errorThrown    : " + errorThrown.message);
        });
    });


    // start model training
    $('#module--TRAIN--run').click(function() {
        $.ajax({
            type: 'POST',
            url: '/module/' + module + '/TRAIN',
            data: $('#module--TRAIN').serialize(),
            dataType: 'json',
            cache: false,
        }).done(function(x, textStatus, jqXHR) {
            refreshModule(true);
        }).fail(function(XMLHttpRequest, textStatus, errorThrown) {
            console.log("XMLHttpRequest : " + XMLHttpRequest.status);
            console.log("textStatus     : " + textStatus);
            console.log("errorThrown    : " + errorThrown.message);
        });
    });
 
   
    // start inference
    $('#module--INFERENCE--run').click(function() {
        $.ajax({
            type: 'POST',
            url: '/module/' + module + '/INFERENCE',
            data: $('#module--INFERENCE').serialize(),
            dataType: 'json',
            cache: false,
        }).done(function(x, textStatus, jqXHR) {
            refreshModule(true);
        }).fail(function(XMLHttpRequest, textStatus, errorThrown) {
            console.log("XMLHttpRequest : " + XMLHttpRequest.status);
            console.log("textStatus     : " + textStatus);
            console.log("errorThrown    : " + errorThrown.message);
        });
    });


    // force stop running job
    $('#force-stop').on('click', function() {
        $('#force-stop').attr('disabled', true).text('STOPPING ...');
        $.ajax({
            type: 'POST',
            url: '/app/interrupt',
            cache: false,
        }).done(function(x, textStatus, jqXHR) {
            sleep(10000); // wait for server stop the thread jobs and output log messages
            refreshModule(true);
        }).fail(function(XMLHttpRequest, textStatus, errorThrown) {
            console.log("XMLHttpRequest : " + XMLHttpRequest.status);
            console.log("textStatus     : " + textStatus);
            console.log("errorThrown    : " + errorThrown.message);
        });
    });

    
    
    // disable config field if customized-architecture is selected
    $('#module--BASE--architecture').on('change', function() {
        refreshConfigField();
    });
    
    
    // open sub-window to select file/folder when clicking Select buttons
    $('button.select-file').on('click', function() {
        let _ = $(this).val().split('+');
        focusedInputField = _[0];
        let selectType = _[1];
        let btn_id = $(this).attr('id');

        $('#filetree').fileTree();

        if (selectType === 'any') {
            selectType = 'folder';
            if ($('#module--TRAIN--trainannfmt').val() === 'COCO' && btn_id === 'module--TRAIN--trainann--button') {
                selectType = 'file';
            }
            if ($('#module--TRAIN--validannfmt').val() === 'COCO' && btn_id === 'module--TRAIN--validann--button') {
                selectType = 'file';
            }
        if ($('#module--TRAIN--testannfmt').val() === 'COCO' && btn_id === 'module--TRAIN--testann--button') {
                selectType = 'file';
            }
        }
        
        $('#filetree-required-filetype').val(selectType);
        if (selectType === 'file') {
            $('#filetree-select-msgbox').text(msgSelectFile);
        } else {
            $('#filetree-select-msgbox').text(msgSelectFolder);
        }
        $('#filetree-selected').val($('#' + focusedInputField).val());
        $('#select-file-popup').fadeIn();
        
    });
    // close select-file-popup window
    $('body').on('click', '#select-file-popup', function(e) {
        if (!$(e.target).closest('#select-file-manager').length) {
            $('#select-file-popup').fadeOut();
        }
    });
    $('body').on('click', '#closeSelectFilePanel', function() {
        $('#select-file-popup').fadeOut();
    });
    // set the selected path
    $('body').on('click', '#selectFile', function() {
        setSelectedFile();
    });

    $('body').on('click', '#filetree', function() {
        if ($('#filetree-required-filetype').val() === $('#filetree-selected-filetype').val()) {
            $('#selectFile').removeClass('button-disable');
            $('#selectFile').attr('disabled', false);
        } else {
            $('#selectFile').addClass('button-disable');
            $('#selectFile').attr('disabled', true);
            if (($('#filetree-required-filetype').val() === 'file') && ($('#filetree-selected-filetype').val() === 'folder')) {
                $('#filetree-select-msgbox').text(msgSelectFile);
            }
            else if (($('#filetree-required-filetype').val() === 'folder') && ($('#filetree-selected-filetype').val() === 'file')) {
                $('#filetree-select-msgbox').text(msgSelectFolder);
            }
        }
        
    });
    // check users whether write file name or not
    $('body').on('keyup keypress blur change', '#filetree-selected', function(e) {
        if ($('#filetree-required-filetype').val() === 'file' && $('#filetree-selected-filetype').val() === 'folder') {
            if (($(this).data('val') === $(this).val()) | ($(this).val().slice(-1)) === '/') {
                $('#selectFile').addClass('button-disable');
                $('#selectFile').attr('disabled', true);
            } else {
                $('#selectFile').removeClass('button-disable');
                $('#selectFile').attr('disabled', false);
                if (e.keyCode !== undefined && e.keyCode === 13) {
                    setSelectedFile();
                }
            }
        }
    });


    $('body').on('click', '#config-edition-popup', function(e) {
        if (!$(e.target).closest('#config-edition-manager').length) {
            $('#config-edition-popup').fadeOut();
        }
    });
    $('body').on('click', '#config-edition-close', function() {
        $('#config-edition-popup').fadeOut();
    });
    $('body').on('click', '#config-edition-save', function() {
        $.ajax({
            type: 'POST',
            url: '/api/modelconfig',
            data: {
                file_path: $('#module--BASE--config').val(), 
                data: $('#config-edition-content').val()
            },
            dataType: 'json',
            cache: false,
        }).done(function(config_content, textStatus, jqXHR) {
            $('#config-edition-msgbox').fadeOut(function() {
                $(this).text(config_content['status']).fadeIn();
            });
            $('#config-edition-content').fadeOut(function() {
                $(this).val(config_content['data']).fadeIn();
            });
            refreshRunningLog();
        }).fail(function(XMLHttpRequest, textStatus, errorThrown) {
                console.log("XMLHttpRequest : " + XMLHttpRequest.status);
                console.log("textStatus     : " + textStatus);
                console.log("errorThrown    : " + errorThrown.message);
        });
    });


    $('#module--TRAIN--strategy').on('change', function() {
        if ($(this).val() === 'resizing') {
            $('#module--TRAIN--resizing-option').css('visibility', '').css('visibility', 'hidden');
        } else {
            $('#module--TRAIN--resizing-option').css('visibility', '').css('visibility', 'visibile');
        }
    });


    $('#module--INFERENCE--strategy').on('change', function() {
        if ($(this).val() === 'resizing') {
            $('#module--INFERENCE--resizing-option').css('visibility', '').css('visibility', 'hidden');
        } else {
            $('#module--INFERENCE--resizing-option').css('visibility', '').css('visibility', 'visibile');
        }
    });

    $('#module--TRAIN--validimages,#module--TRAIN--validann').on({
        'mouseenter': function() {
            $(this).attr('placeholder', msgLeftBank.replace('____', 'validation'));
        },
        'mouseleave': function() {
            $(this).removeAttr('placeholder');
        }
    });
    $('#module--TRAIN--testimages,#module--TRAIN--testann').on({
        'mouseenter': function() {
            $(this).attr('placeholder', msgLeftBank.replace('____', 'test'));
        },
        'mouseleave': function() {
            $(this).removeAttr('placeholder');
        }
    });

    $('#filetree-2').fileTree();
     
});

   
// check the running status and update logs
refreshModule(true);
window.setInterval(function() {
    refreshModule();
}, 5000);



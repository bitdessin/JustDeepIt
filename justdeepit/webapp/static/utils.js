var moduleRunningStatus = {'mode': null, 'status': null, 'timestamp': null};
var focusedInputField = null;
var msgSelectFile = 'Please select a FILE or add a file name after the selected folder.';
var msgSelectFolder = 'Please select a FOLDER';
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
        url: '/api/params',
        data: {module: module},
        dataType: 'json',
        cache: false,
    }).done(function(formFields, textStatus, jqXHR) {
        for (let formId in formFields) {
            if (formId !== 'module') {
                for (let fieldId in formFields[formId]) {
                    let fieldId_ = '#module-' + formId + '-' + fieldId;
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
        if ($('#module-config-status').val() !== 'COMPLETED') {
            tabClasses = ['module-tab-active', 'module-tab-disabled', 'module-tab-disabled'];
        } else {
            tabClasses = ['module-tab-active', null, null];
        }
    } else {
        if (moduleRunningStatus.status === 'RUNNING') {
            if (moduleRunningStatus.mode === 'config') {
                tabClasses = ['module-tab-active-disabled', 'module-tab-disabled', 'module-tab-disabled'];
            }
            else if (moduleRunningStatus.mode === 'training') {
                tabClasses = ['module-tab-disabled', 'module-tab-active-disabled', 'module-tab-disabled'];
            }
            else if (moduleRunningStatus.mode === 'inference') {
                tabClasses = ['module-tab-disabled', 'module-tab-disabled', 'module-tab-active-disabled'];
            }
        } else {
            if (moduleRunningStatus.mode === 'config') {
                tabClasses = ['module-tab-active', null, null];
            }
            else if (moduleRunningStatus.mode === 'training') {
                tabClasses = [null, 'module-tab-active', null];
            }
            else if (moduleRunningStatus.mode === 'inference') {
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
    if (focusedInputField == 'module-training-model_weight') {
        if (selectedFilePath.slice((selectedFilePath.lastIndexOf(".") - 1 >>> 0) + 2) !== 'pth') {
            selectedFilePath = selectedFilePath + '.pth';
        }
    }
    console.log(selectedFilePath);
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
    let forms = ['#module-config', '#module-training', '#module-inference'];

    for (let formId of forms) {

        let buttonId = formId + '-run';
        let isValid = true;
        $(formId + ' .file-path').each(function() {
            required_field = true;
            if ($(this).attr('id') === 'module-config-config') required_field = false;

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
        if (formId == '#module-config') {
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
    if ($('#module-config-architecture').val() != 'custom') {
        $('#module-config-config').val(null);
        $('#module-config-config').attr('readonly', true);
        $('#config-select-open').attr('disabled', true);
        $('#config-select-open').addClass('button-disable');
    } else {
        $('#module-config-config').attr('readonly', false);
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
        if ($('#module-config-backend').val().toLowerCase() == 'mmdetection') {
            default_optmizer = 'dict(type="Adam", lr=0.001)';
            default_scheduler = 'dict(policy="CosineAnnealing", warmup="linear", warmup_iters=1000, warmup_ratio=0.1)';
        }
    }
    $('#module-training-optimizer').attr('placeholder', default_optmizer);
    $('#module-training-scheduler').attr('placeholder', default_scheduler);
    
    if ($('#module').val() !== 'SOD') {
        if ($('#module-config-backend').val().toLowerCase() === 'detectron2') {
            $('#module-training-optimizer').attr('disabled', true);
            $('#module-training-scheduler').attr('disabled', true);
        } else {
            $('#module-training-optimizer').attr('disabled', false);
            $('#module-training-scheduler').attr('disabled', false);
        }
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
    $('#module-config-run').click(function() {
        $.ajax({
            type: 'POST',
            url: '/module/' + module + '/config',
            data: $('#module-config').serialize(),
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
    $('#module-training-run').click(function() {
        $.ajax({
            type: 'POST',
            url: '/module/' + module + '/training',
            data: $('#module-training').serialize(),
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
    $('#module-inference-run').click(function() {
        $.ajax({
            type: 'POST',
            url: '/module/' + module + '/inference',
            data: $('#module-inference').serialize(),
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



    // reload NN available architectures according to the backend
    $('#module-config-backend').on('change', function() {
        $.ajax({
            type: 'GET',
            url: '/api/architecture',
            data: {
                module: module,
                backend: $('#module-config-backend').val(),
            },
            dataType: 'json',
            cache: false,
        }).done(function(arch, textStatus, jqXHR) {
            let archListHTML = '';
            for (let i = 0; i < arch.length; i++) {
                archListHTML = archListHTML + '<option value="' + arch[i] + '">' + arch[i] + '</option>';
            }
            $('#module-config-architecture').html(archListHTML);
            refreshConfigField();
            refreshTrainOptimizer();
        }).fail(function(XMLHttpRequest, textStatus, errorThrown) {
            console.log("XMLHttpRequest : " + XMLHttpRequest.status);
            console.log("textStatus     : " + textStatus);
            console.log("errorThrown    : " + errorThrown.message);
        });
    });
    
    
    // disable config field if customized-architecture is selected
    $('#module-config-architecture').on('change', function() {
        refreshConfigField();
    });
    
    
    // open sub-window to select file/folder when clicking Select buttons
    $('button.select-file').on('click', function() {
        let _ = $(this).val().split('+');
        focusedInputField = _[0];
        let selectType = _[1];
        $.ajax({
            type: 'GET',
            url: '/api/dirtree',
            dataType: 'json',
            cache: false,
        }).done(function(dirtree, textStatus, jqXHR) {
            $('#filetree').tree('destroy');
            $('#filetree').tree({
                    openedIcon: '&#x0229F;',
                    closedIcon: '&#x0229E;',
                    showEmptyFolder: true,
                    autoOpen: 0,
                    selectable: true,
                    data: dirtree,
            });
            if (selectType === 'any') {
                if ($('#module-training-annotation_format').val() === 'COCO') {
                    selectType = 'file';
                } else {
                    selectType = 'folder';
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
        }).fail(function(XMLHttpRequest, textStatus, errorThrown) {
                console.log("XMLHttpRequest : " + XMLHttpRequest.status);
                console.log("textStatus     : " + textStatus);
                console.log("errorThrown    : " + errorThrown.message);
        });
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
        // wait for JqTree to set up the selected file
        setTimeout(function() {
            $('#filetree-selected').val(node2path($('#filetree').tree('getSelectedNode')).slice(1));
            $('#filetree-selected').data('val', node2path($('#filetree').tree('getSelectedNode')).slice(1));
            $('#filetree-selected-filetype').val(node2filetype($('#filetree').tree('getSelectedNode')));

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
        }, 300);
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

    
    /*
    $('button#config-edition-open').on('click', function() {
        $.ajax({
            type: 'GET',
            url: '/api/modelconfig',
            data: {config_fpath: $('#module-config-config').val()},
            dataType: 'json',
            cache: false,
        }).done(function(config_content, textStatus, jqXHR) {
            $('#config-edition-msgbox').text(config_content['status']);
            $('#config-edition-content').val(config_content['data']);
            $('#config-edition-popup').fadeIn();
        }).fail(function(XMLHttpRequest, textStatus, errorThrown) {
                console.log("XMLHttpRequest : " + XMLHttpRequest.status);
                console.log("textStatus     : " + textStatus);
                console.log("errorThrown    : " + errorThrown.message);
        });
    });
    */

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
                file_path: $('#module-config-config').val(), 
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


    $('#module-training-strategy').on('change', function() {
        if ($(this).val() === 'resizing') {
            $('#module-training-resizing-option').css('visibility', '').css('visibility', 'hidden');
        } else {
            $('#module-training-resizing-option').css('visibility', '').css('visibility', 'visibile');
        }
    });


    $('#module-inference-strategy').on('change', function() {
        if ($(this).val() === 'resizing') {
            $('#module-inference-resizing-option').css('visibility', '').css('visibility', 'hidden');
        } else {
            $('#module-inference-resizing-option').css('visibility', '').css('visibility', 'visibile');
        }
    });
    
});
 
   
// check the running status and update logs
refreshModule(true);
window.setInterval(function() {
    refreshModule();
}, 5000);



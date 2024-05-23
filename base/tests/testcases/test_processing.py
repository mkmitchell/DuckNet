BaseCase = __import__('base_case').BaseCase
import pytest, os


class ProcessingTest(BaseCase):
    @pytest.mark.slow
    def test_basic_processing_success(self):
        self.open_main(static=False)
        
        self.send_input_files_from_assets([
            "test_image0.jpg",
            "test_image1.jpg",
            "test_image2.tiff",
        ])

        #open one file
        self.click('label:contains("test_image1.jpg")')
        root_css = '[filename="test_image1.jpg"]'
        #click on the processing button
        self.click(root_css+' .play.icon')

        #a dimmer that indicates that processing is in progress should be visible
        self.wait_for_element_visible(root_css+' .dimmer', timeout=2)
        #there should be no error indication
        assert not self.find_visible_elements('.error, .failed')
        #dimmer should not be closable with a click
        self.execute_script(f'''$('{root_css} .dimmer').click()''') #self.click doesnt work
        self.sleep(0.5)

        self.assert_element_visible(root_css+' .dimmer', timeout=0.5)

        #after processing is done, the dimmer should be gone (can take a while)
        self.wait_for_element_not_visible(root_css+' .dimmer', timeout=6)
        #make sure the row label is bold to indicate that this file is processed
        script = f''' return $('{root_css} label').css('font-weight') '''
        assert int(self.execute_script(script)) > 500

        #result image is grayed out with contrast(0) at the start, this should be removed now
        script = f''' return $('{root_css} .result img').css('filter') '''
        if self.is_element_visible(f'{root_css} .result img'):
            assert 'contrast(0)' not in self.execute_script(script)

        #TODO: check some indication that a file is finished processing (bold font in the file list)
        
        if self.demo_mode:
            self.sleep(1)
        
    
    def test_basic_upload(self):
        self.open_main(static=False)
        
        self.send_input_files_from_assets(["test_image1.jpg"])
        script = '''upload_file_to_flask(GLOBAL.files["test_image1.jpg"]).done(arguments[0]())'''
        self.execute_async_script(script)
        self.sleep(1)
        self.assert_link_status_code_is_not_404(f"http://localhost:{self.port}/images/test_image1.jpg")
        self.assert_no_404_errors()

        if self.demo_mode:
            self.sleep(1)


    def test_upload_failed(self):
        self.open_main(static=True)  #static so that upload fails
        
        self.send_input_files_from_assets([
            "test_image0.jpg",
            "test_image1.jpg",
            "test_image2.tiff",
        ])

        #open one file
        self.click('label:contains("test_image1.jpg")')
        root_css = '[filename="test_image1.jpg"]'
        #click on the processing button
        self.click(root_css+' .play.icon')

        #should fail because static file is opened
        #self.wait_for_element_visible('.error', timeout=3)
        self.sleep(1)
        assert self.execute_script('''return $('.dimmer :contains("failed"):visible').length''') > 0
        #self.wait_for_element_visible('.dimmer :contains("failed")', timeout=3) #doesnt work

        #dimmer should hide after click
        self.execute_script(f'''$('.dimmer:visible').click()''') #self.click doesnt work
        self.wait_for_element_not_visible(root_css+' .dimmer', timeout=6)

        #download button should be disabled
        if self.is_element_visible(root_css + ' a.download'):
            assert 'disabled' in self.get_attribute(root_css + ' a.download', 'class')
        #processing button should be enabled
        assert 'disabled' not in self.get_attribute(root_css + ' a.process', 'class')

        if self.demo_mode:
            self.sleep(1)
